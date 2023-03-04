import hydra
from omegaconf import DictConfig, OmegaConf
import datetime
import os
import random

# suppress D4RL warnings
os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"

import gym
import numpy as np
import torch
import wandb

from ppo.ppo_algo import PPO
from ppo.replay_buffer import RolloutBuffer
from hca.hca_model import HCAModel
from hca.hca_buffer import HCABuffer, calculate_mc_returns

from wrappers.lorl import TASKS
from utils import (
    get_hindsight_logprobs,
    get_env,
)
from eval import eval


def train(args):
    # Environment
    env = get_env(args)

    if isinstance(env.action_space, gym.spaces.Box):
        continuous = True
    else:
        continuous = False

    if continuous:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    if isinstance(env.observation_space, gym.spaces.Dict):
        # gridworld env
        state_dim = env.observation_space["map"].shape[0] + 1
    else:
        state_dim = env.observation_space.shape[0]

    if args.training.seed:
        print(
            "============================================================================================"
        )
        print(f"Setting seed: {args.training.seed}")
        print(
            "============================================================================================"
        )
        env.seed(args.training.seed)
        env.action_space.seed(args.training.seed)
        random.seed(args.training.seed)
        torch.manual_seed(args.training.seed)
        np.random.seed(args.training.seed)

    reward_type = "sparse" if args.env.sparse else "dense"
    exp_name = f"{args.agent.name}_{reward_type}_{args.env.name}"
    if args.env.type == "gridworld":
        exp_name += f":{args.env.puzzle_path.lstrip('maps/').rstrip('.txt')}"
    if args.training.exp_name_modifier:
        exp_name += "_" + args.training.exp_name_modifier

    # Device
    device = torch.device(args.training.device)

    if args.training.save_model_freq:
        checkpoint_path = f"../checkpoints/{exp_name}_"
        checkpoint_path += f"{datetime.datetime.now().replace(microsecond=0)}"
        setattr(args, "savedir", checkpoint_path)
        os.makedirs(checkpoint_path, exist_ok=True)

    # Wandb Initialization
    if args.training.wandb:
        wandb.init(
            name=exp_name,
            project=args.env.name,
            config=vars(args),
            entity="ca-exploration",
        )

    # Agent
    agent = PPO(state_dim, action_dim, args.agent.lr, continuous, device, args)

    if args.training.checkpoint:
        checkpoint = torch.load(args.training.checkpoint)
        agent.load(checkpoint["policy"])

    # HCA model
    h_model = None
    if args.agent.name == "ppo-hca":
        h_model = HCAModel(
            state_dim + 1,  # this is for return-conditioned
            action_dim,
            continuous=continuous,
            n_layers=args.agent.hca_n_layers,
            hidden_size=args.agent.hca_hidden_size,
            activation_fn=args.agent.hca_activation,
            dropout_p=args.agent.hca_dropout,
            batch_size=args.agent.hca_batchsize,
            lr=args.agent.hca_lr,
            device=args.training.device,
        )
        h_model = h_model.to(args.training.device)
        if args.agent.hca_checkpoint:
            hca_checkpoint = torch.load(args.agent.hca_checkpoint)
            h_model.load(hca_checkpoint["model"], strict=True)
            print(
                f"Successfully loaded hca model from {args.aget.hca_checkpoint}!"
            )

        # HCA Buffer
        if continuous:
            hca_buffer = HCABuffer(
                exp_name,
                action_dim=action_dim,
                train_val_split=args.agent.hca_train_val_split,
            )
        else:
            hca_buffer = HCABuffer(
                exp_name,
                action_dim=1,
                train_val_split=args.agent.hca_train_val_split,
            )

    # Replay Buffer for PPO
    buffer = RolloutBuffer()

    # logging
    total_rewards, total_successes = [], []
    total_losses, action_losses, value_losses, entropies = [], [], [], []
    hca_ratio_mins, hca_ratio_maxes, hca_ratio_means, hca_ratio_stds = (
        [],
        [],
        [],
        [],
    )

    # track total training time
    start_time = datetime.datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print(
        "============================================================================================"
    )

    num_total_steps = 0
    steps_between_logs = 0
    steps_between_evals = 0
    episodes_collected = 0

    state = env.reset()
    done = False
    current_ep_reward = 0
    current_ep_length = 0

    while num_total_steps <= args.training.max_training_env_steps:

        # Exploration:
        for t in range(1, args.agent.env_steps_per_update + 1):
            if args.agent.name == "random":
                action = env.action_space.sample()
                action_logprob = None
                value = None
            else:
                # select action with policy
                action, action_logprob, value = agent.select_action(state)
                if continuous:
                    action = action.numpy().flatten()
                    action = action.clip(
                        env.action_space.low, env.action_space.high
                    )
                else:
                    action = action.item()

            # step env
            next_state, reward, done, info = env.step(action)
            current_ep_reward += reward
            num_total_steps += 1
            current_ep_length += 1
            steps_between_logs += 1
            steps_between_evals += 1

            # determine if episode is done, and if it is because of terminal state or timeout.
            if done:
                # trajectory didn't reach a terminal state; bootstrap value target.
                if not info["terminal_state"]:
                    _, _, next_value = agent.select_action(next_state)
                else:
                    next_value = 0.0

                # Add in the final value to the reward to account for potentially not reaching terminal state.
                reward += agent.gamma * next_value
                total_rewards.append(current_ep_reward)
                total_successes.append(info.get("success", 0.0))

                # reset env and trackers.
                next_state = env.reset()
                episodes_collected += 1
                current_ep_reward = 0.0
                current_ep_length = 0


            # add transition to buffer
            buffer.states.append(state)
            buffer.actions.append(action)
            buffer.logprobs.append(action_logprob)
            buffer.values.append(value)
            buffer.rewards.append(reward)
            buffer.terminals.append(done)

            state = next_state

        # Batch has been collected; compute the last value if needed, and put it in buffer.
        if not done:
            if not info["terminal_state"]:
                _, _, final_value = agent.select_action(state)
                buffer.rewards[-1] += agent.gamma * final_value

        # TODO: Credit assignment.

        # Policy update (PPO)
        if args.agent.name != "random":
            total_loss, action_loss, value_loss, entropy, hca_ratio_dict = agent.update(buffer)

            total_losses.append(total_loss)
            action_losses.append(action_loss)
            value_losses.append(value_loss)
            entropies.append(entropy)
            if hca_ratio_dict:
                hca_ratio_mins.append(hca_ratio_dict["min"])
                hca_ratio_maxes.append(hca_ratio_dict["max"])
                hca_ratio_means.append(hca_ratio_dict["mean"])
                hca_ratio_stds.append(hca_ratio_dict["std"])

            buffer.clear()

        # logging
        if args.training.log_freq and steps_between_logs >= args.training.log_freq:
            steps_between_logs = 0

            avg_reward = np.mean(total_rewards)
            avg_success = np.mean(total_successes)

            hca_ratio_min = (
                np.mean(hca_ratio_mins) if len(hca_ratio_mins) > 0 else 0.0
            )
            hca_ratio_max = (
                np.mean(hca_ratio_maxes) if len(hca_ratio_maxes) > 0 else 0.0
            )
            hca_ratio_mean = (
                np.mean(hca_ratio_means) if len(hca_ratio_means) > 0 else 0.0
            )
            hca_ratio_std = (
                np.mean(hca_ratio_stds) if len(hca_ratio_stds) > 0 else 0.0
            )

            if args.training.wandb:
                wandb.log(
                    {
                        "training/avg_rewards": avg_reward,
                        "training/avg_success": avg_success,
                        "training/total_loss": np.mean(total_losses),
                        "training/action_loss": np.mean(action_losses),
                        "training/value_loss": np.mean(value_losses),
                        "training/entropy": np.mean(entropies),
                    },
                    step=num_total_steps,
                )
                if args.agent.name == "ppo-hca":
                    wandb.log(
                        {
                            "training/hca_ratio_min": hca_ratio_min,
                            "training/hca_ratio_max": hca_ratio_max,
                            "training/hca_ratio_mean": hca_ratio_mean,
                            "training/hca_ratio_std": hca_ratio_std,
                        },
                        step=num_total_steps,
                    )

            print(
                f"Steps: {num_total_steps} \t\t Episodes: {episodes_collected} \t\t Average Reward: {avg_reward:.4f} \t\t Average Success: {avg_success:.4f}"
            )

            total_rewards, total_successes = [], []
            total_losses, action_losses, value_losses, entropies = (
                [],
                [],
                [],
                [],
            )

        # save model weights
        if (
                args.training.save_model_freq
                and episodes_collected % args.training.save_model_freq == 0
        ):
            print(
                "--------------------------------------------------------------------------------------------"
            )
            print("saving model at : " + checkpoint_path)
            agent.save(f"{checkpoint_path}/model_{episodes_collected}.pt", vars(args))
            print("model saved")
            print(
                "Elapsed Time  : ",
                datetime.datetime.now().replace(microsecond=0) - start_time,
            )
            print(
                "--------------------------------------------------------------------------------------------"
            )

        if args.training.eval_freq and steps_between_evals >= args.training.eval_freq:
            steps_between_evals = 0
            eval_avg_reward, eval_avg_success = eval(env, agent, args)

            if args.training.wandb:
                wandb.log(
                    {
                        "eval/avg_rewards": eval_avg_reward,
                        "eval/avg_success": eval_avg_success,
                    },
                    step=num_total_steps,
                )

    ## SAVE MODELS
    if args.training.save_model_freq:
        print(
            "--------------------------------------------------------------------------------------------"
        )
        print("Final Checkpoint Save!!")
        print("saving model at : " + checkpoint_path)
        agent.save(f"{checkpoint_path}/model_{num_total_steps}.pt", vars(args))
        print("model saved")
        print(
            "Elapsed Time  : ",
            datetime.datetime.now().replace(microsecond=0) - start_time,
        )
        print(
            "--------------------------------------------------------------------------------------------"
        )

def get_args(cfg: DictConfig):
    cfg.training.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg.hydra_base_dir = os.getcwd()
    return cfg


@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    args = get_args(cfg)

    print("--> Running in ", os.getcwd())

    # train
    print(OmegaConf.to_yaml(cfg))
    train(args)


if __name__ == "__main__":
    main()
