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

from ppo.ppo_algo import PPO
from ppo.replay_buffer import RolloutBuffer
from hca.hca_model import HCAModel
from hca.hca_buffer import HCABuffer, calculate_mc_returns

from utils import (
    get_hindsight_logprobs,
    get_env,
)
from eval import eval
from logger import PPO_Stats, HCA_Stats, Logger


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
    if args.logger.exp_name_modifier:
        exp_name += "_" + args.logger.exp_name_modifier

    # Device
    device = torch.device(args.training.device)

    if args.training.save_model_freq:
        checkpoint_path = f"../checkpoints/{exp_name}_"
        checkpoint_path += f"{datetime.datetime.now().replace(microsecond=0)}"
        setattr(args, "savedir", checkpoint_path)
        os.makedirs(checkpoint_path, exist_ok=True)

    # Initialize Logger
    if args.training.log_freq:
        logger = Logger(
            exp_name,
            args.env.name,
            args.agent.name,
            vars(args),
            "ca-exploration",
            args.logger.wandb,
            group_modifier=args.logger.group_name_modifier,
        )

    # Agent
    agent = PPO(state_dim, action_dim, args.agent.lr, continuous, device, args)

    if args.training.checkpoint:
        checkpoint = torch.load(args.training.checkpoint)
        agent.load(checkpoint["policy"])

    # HCA model
    h_model = None
    if args.agent.name in ["ppo-hca", "hca-gamma"]:
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

    # initial eval
    print(" ============ Evaluating =============")
    eval_avg_reward, eval_avg_success = eval(env, agent, args)
    logger.log(
        {
            "avg_rewards": eval_avg_reward,
            "avg_success": eval_avg_success,
        },
        step=0,
        wandb_prefix="eval",
    )
    print("======= Finished Evaluating =========")

    for episode in range(1, args.training.max_training_episodes + 1):
        state = env.reset()

        current_ep_reward = 0
        done = False

        states, actions, logprobs, rewards, terminals = [], [], [], [], []

        # Exploration
        while not done:
            if args.agent.name == "random":
                action = env.action_space.sample()
                action_logprob = None
            else:
                # select action with policy
                action, action_logprob = agent.select_action(state)
                if continuous:
                    action = action.numpy().flatten()
                    clipped_action = action.clip(
                        env.action_space.low, env.action_space.high
                    )
                else:
                    action = action.item()
                    clipped_action = action
            states.append(state)
            actions.append(action)
            logprobs.append(action_logprob)

            # Step in env
            state, reward, done, info = env.step(clipped_action)

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

        if args.agent.name in ["ppo-hca", "hca-gamma"]:
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

        # if episode >= 20000:
        #     buffer.hindsight_logprobs.append(hindsight_logprobs)
        # else:
        #     buffer.hindsight_logprobs.append(
        #         [np.exp(x) for x in hindsight_logprobs]
        #     )

        if args.agent.name in ["ppo-hca", "hca-gamma"]:
            hca_buffer.add_episode(states, actions, rewards, agent.gamma)

        # Assign credit
        if args.agent.name in ["ppo-hca", "hca-gamma"] and (
            episode % args.agent.hca_update_every == 0
            or episode == args.agent.update_every
        ):
            # always update the HCA model the first time before a PPO update
            # reset the model if you want
            if args.agent.refresh_hca:
                h_model.reset_parameters()

            # update the HCA model
            for _ in range(args.agent.hca_epochs):
                hca_results = h_model.update(hca_buffer)
            # Clear the HCA buffer
            hca_buffer.clear()

            # Log every time we update the model and don't use the log freq
            hca_stats = HCA_Stats(**hca_results)

            print(" ============ Updated HCA model =============")
            logger.log(hca_stats, step=episode, wandb_prefix="training")
            print("=============================================")

        # Agent update (PPO)
        if (
            args.agent.name != "random"
            and episode % args.agent.update_every == 0
        ):
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

            stats = PPO_Stats(
                avg_rewards=avg_reward,
                avg_success=avg_success,
                total_loss=np.mean(total_losses),
                action_loss=np.mean(action_losses),
                value_loss=np.mean(value_losses),
                entropy=np.mean(entropies),
                hca_ratio_max=hca_ratio_max,
                hca_ratio_min=hca_ratio_min,
                hca_ratio_mean=hca_ratio_mean,
                hca_ratio_std=hca_ratio_std,
            )

            logger.log(stats, step=episode, wandb_prefix="training")

            total_rewards, total_successes = [], []
            total_losses, action_losses, value_losses, entropies = (
                [],
                [],
                [],
                [],
            )

            hca_ratio_mins, hca_ratio_maxes, hca_ratio_means, hca_ratio_stds = (
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

        if args.training.eval_freq and episode % args.training.eval_freq == 0:
            eval_avg_reward, eval_avg_success = eval(env, agent, args)

            print(" ============ Evaluating =============")
            logger.log(
                {
                    "avg_rewards": eval_avg_reward,
                    "avg_success": eval_avg_success,
                },
                step=episode,
                wandb_prefix="eval",
            )
            print("======= Finished Evaluating =========")

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


@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    args = get_args(cfg)

    print("--> Running in ", os.getcwd())

    # train
    print(OmegaConf.to_yaml(cfg))
    train(args)


if __name__ == "__main__":
    main()
