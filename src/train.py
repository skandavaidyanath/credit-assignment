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

from lorl import TASKS
from utils import (
    get_hindsight_logprobs,
    HCABuffer,
    calculate_mc_returns,
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
        exp_name += (
            f":{args.training.puzzle_path.lstrip('maps/').rstrip('.txt')}"
        )
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
    hca_buffer_online = None
    h_model = None
    hca_opt = None
    if args.agent.name == "ppo-hca":
        if args.agent.hca_checkpoint:
            hca_checkpoint = torch.load(args.agent.hca_checkpoint)
            h_model = HCAModel(
                state_dim + 1,
                action_dim,
                continuous=continuous,
                n_layers=hca_checkpoint["args"]["n_layers"],
                hidden_size=hca_checkpoint["args"]["hidden_size"],
            )
            h_model.load(hca_checkpoint["model"])
            print("successfully loaded hca model!")
        else:
            assert (
                args.agent.update_hca_online
            ), "If not specifying a HCA checkpoint, use --update-hca-online"
            h_model = HCAModel(
                state_dim + 1,
                action_dim,
                continuous=continuous,
                n_layers=args.agent.hca_n_layers,
                hidden_size=args.agent.hca_hidden_size,
            )
        if args.agent.update_hca_online:
            if continuous:
                hca_buffer_online = HCABuffer(exp_name, action_dim=action_dim)
            else:
                hca_buffer_online = HCABuffer(exp_name, action_dim=1)
            hca_opt = torch.optim.Adam(h_model.parameters(), lr=args.hca_lr)

    # Replay Memory
    buffer = RolloutBuffer()

    hca_buffer = None
    if args.training.collect_hca_data:
        if continuous:
            hca_buffer = HCABuffer(exp_name, action_dim=action_dim)
        else:
            hca_buffer = HCABuffer(exp_name, action_dim=1)

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

    for episode in range(1, args.training.max_training_episodes + 1):
        if args.env.type == "lorl":
            state = env.reset(args.env.task)
        else:
            state = env.reset()

        current_ep_reward = 0
        done = False

        states, actions, logprobs, rewards, terminals = [], [], [], [], []

        while not done:
            if args.agent.name == "random":
                action = env.action_space.sample()
                action_logprob = None
            else:
                # select action with policy
                action, action_logprob = agent.select_action(state)
                if continuous:
                    action = action.numpy().flatten()
                    action = action.clip(
                        env.action_space.low, env.action_space.high
                    )
                else:
                    action = action.item()
            states.append(state)
            actions.append(action)
            logprobs.append(action_logprob)

            # Step in env
            state, reward, done, info = env.step(action)

            # saving reward and terminals
            rewards.append(float(reward))
            terminals.append(done)

            current_ep_reward += reward

        total_rewards.append(current_ep_reward)
        if "success" in info:
            total_successes.append(info["success"])
        else:
            total_successes.append(0.0)

        hindsight_logprobs = []
        if args.agent.name == "ppo-hca":
            returns = calculate_mc_returns(rewards, terminals, agent.gamma)
            hindsight_logprobs = get_hindsight_logprobs(
                h_model, states, returns, actions
            )

        buffer.states.append(states)
        buffer.actions.append(actions)
        buffer.logprobs.append(logprobs)
        buffer.rewards.append(rewards)
        buffer.terminals.append(terminals)
        buffer.hindsight_logprobs.append(hindsight_logprobs)

        if hca_buffer_online:
            hca_buffer_online.add_episode(states, actions, rewards, agent.gamma)

        # update the HCA function, if applicable
        mean_hca_loss = 0.0
        if args.agent.update_hca_online and (
            episode % args.agent.hca_update_every == 0
            or episode == args.agent.update_every
        ):
            assert hca_buffer_online is not None
            assert h_model is not None
            hca_losses = []
            for _ in range(args.agent.hca_num_updates):
                X, y = hca_buffer_online.get_batch(args.agent.hca_batchsize)
                X, y = torch.from_numpy(X).float(), torch.from_numpy(y)
                hca_loss = h_model.train_step(X, y, hca_opt, device)
                hca_losses.append(hca_loss)
            mean_hca_loss = np.mean(hca_losses)

        # update PPO agent
        if not args.agent.name == "random" and episode % args.agent.update_every == 0:
            (
                total_loss,
                action_loss,
                value_loss,
                entropy,
                hca_ratio_dict,
            ) = agent.update(buffer)
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

        # store data for hindsight function training
        if args.training.collect_hca_data:
            hca_buffer.add_episode(states, actions, rewards, agent.gamma)

        # logging
        if args.training.log_freq and episode % args.training.log_freq == 0:
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
                    step=episode,
                )
                if args.agent.name == "ppo-hca":
                    wandb.log(
                        {
                            "training/hca_loss": mean_hca_loss,
                            "training/hca_ratio_min": hca_ratio_min,
                            "training/hca_ratio_max": hca_ratio_max,
                            "training/hca_ratio_mean": hca_ratio_mean,
                            "training/hca_ratio_std": hca_ratio_std,
                        },
                        step=episode,
                    )

            print(
                f"Episode: {episode} \t\t Average Reward: {avg_reward:.4f} \t\t Average Success: {avg_success:.4f}"
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
            and episode % args.training.save_model_freq == 0
        ):
            print(
                "--------------------------------------------------------------------------------------------"
            )
            print("saving model at : " + checkpoint_path)
            agent.save(f"{checkpoint_path}/model_{episode}.pt", vars(args))
            print("model saved")
            print(
                "Elapsed Time  : ",
                datetime.datetime.now().replace(microsecond=0) - start_time,
            )
            print(
                "--------------------------------------------------------------------------------------------"
            )

        if (
            hca_buffer
            and args.training.hca_data_save_freq
            and episode % args.training.hca_data_save_freq == 0
        ):
            hca_buffer.save_data(action_dim)

        if args.training.eval_freq and episode % args.training.eval_freq == 0:
            eval_avg_reward, eval_avg_success = eval(env, agent, args)

            if args.training.wandb:
                wandb.log(
                    {
                        "eval/avg_rewards": eval_avg_reward,
                        "eval/avg_success": eval_avg_success,
                    },
                    step=episode,
                )

    ## SAVE MODELS
    if args.training.save_model_freq:
        print(
            "--------------------------------------------------------------------------------------------"
        )
        print("Final Checkpoint Save!!")
        print("saving model at : " + checkpoint_path)
        agent.save(f"{checkpoint_path}/model_{episode}.pt", vars(args))
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


@hydra.main(config_path="conf", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    args = get_args(cfg)

    print("--> Running in ", os.getcwd())

    # train
    print(OmegaConf.to_yaml(cfg))
    train(args)


if __name__ == "__main__":
    main()