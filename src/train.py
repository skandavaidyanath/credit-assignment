import argparse
import datetime
import os
import gym
import numpy as np
import torch

import utils
import wandb

from gridworld import GridWorld
from eval import eval
from ppo import PPO
from replay_buffer import RolloutBuffer
from utils import get_hindsight_logprobs
from hca_model import HCAModel


def train(args):
    # Environment
    env = GridWorld(args.puzzle_path, sparse=args.sparse)

    if isinstance(env.action_space, gym.spaces.Box):
        continuous = True
    else:
        continuous = False

    if continuous:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    state_dim = env.observation_space.shape[0]

    if args.seed:
        print(
            "============================================================================================"
        )
        print(f"Setting seed: {args.seed}")
        print(
            "============================================================================================"
        )
        env.seed(args.seed)
        env.action_space.seed(args.seed)

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    exp_name = f"{args.method}_{args.env_name}:{args.puzzle_path.lstrip('maps/').rstrip('.txt')}"

    # Device
    device = torch.device(args.device)
    
    if args.save_model_freq:
        checkpoint_path = f"checkpoints/{exp_name}_"
        checkpoint_path += f"{datetime.datetime.now().replace(microsecond=0)}"
        setattr(args, "savedir", checkpoint_path)
        os.makedirs(checkpoint_path, exist_ok=True)

    # Wandb Initialization
    if args.wandb:
        wandb.init(
            name=exp_name,
            project=args.env_name,
            config=vars(args),
            entity="ca-exploration",
        )

    # Agent
    agent = PPO(state_dim, action_dim, args.lr, continuous, device, args)

    if args.method == "ppo-hca":
        hca_checkpoint = torch.load(args.hca_checkpoint)
        h_model = HCAModel(state_dim+1, action_dim, hca_checkpoint["args"]["n_layers"], hca_checkpoint["args"]["hidden_size"])
        h_model.load(hca_checkpoint["model"])

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        agent.load(checkpoint["policy"])

    # Replay Memory
    buffer = RolloutBuffer()

    hca_buffer = None
    if args.collect_hca_data:
        hca_buffer = utils.HCABuffer(exp_name)

    # logging
    total_rewards, total_successes = [], []
    total_losses, action_losses, value_losses, entropies = [], [], [], []

    # track total training time
    start_time = datetime.datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print(
        "============================================================================================"
    )

    for episode in range(1, args.max_training_episodes + 1):
        state = env.reset()
        current_ep_reward = 0
        done = False

        states, actions, logprobs, rewards, terminals = [], [], [], [], []

        while not done:
            # select action with policy
            action, action_logprob = agent.select_action(state)
            if continuous:
                action = action.numpy().flatten()
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
        total_successes.append(info["success"])

        hindsight_logprobs = []
        if args.method == "ppo-hca":
            returns = utils.calculate_mc_returns(rewards, terminals, agent.gamma)
            hindsight_logprobs = get_hindsight_logprobs(h_model, states, returns, actions)

        buffer.states.append(states)
        buffer.actions.append(actions)
        buffer.logprobs.append(logprobs)
        buffer.rewards.append(rewards)
        buffer.terminals.append(terminals)
        buffer.hindsight_logprobs.append(hindsight_logprobs)

        # update PPO agent
        if episode % args.update_every == 0:
            total_loss, action_loss, value_loss, entropy = agent.update(buffer)
            total_losses.append(total_loss)
            action_losses.append(action_loss)
            value_losses.append(value_loss)
            entropies.append(entropy)
            buffer.clear()

        # store data for hindsight function training
        if args.collect_hca_data:
            hca_buffer.add_episode(states, actions, rewards, agent.gamma)

        # logging
        if args.log_freq and episode % args.log_freq == 0:
            avg_reward = np.mean(total_rewards)
            avg_success = np.mean(total_successes)

            if args.wandb:
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
        if args.save_model_freq and episode % args.save_model_freq == 0:
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

        if hca_buffer and args.hca_data_save_freq and episode % args.hca_data_save_freq == 0:
            hca_buffer.save_data(action_dim)

        if args.eval_freq and episode % args.eval_freq == 0:
            eval_avg_reward, eval_avg_success = eval(env, agent)

            if args.wandb:
                wandb.log(
                    {
                        "eval/avg_rewards": eval_avg_reward,
                        "eval/avg_success": eval_avg_success,
                    },
                    step=episode,
                )

    ## SAVE MODELS
    if args.save_model_freq:
        print("--------------------------------------------------------------------------------------------")
        print("Final Checkpoint Save!!")
        print("saving model at : " + checkpoint_path)
        agent.save(f'{checkpoint_path}/model_{episode}.pt', vars(args))
        print("model saved")
        print("Elapsed Time  : ", datetime.datetime.now().replace(microsecond=0) - start_time)
        print("--------------------------------------------------------------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CA Based Exploration RL Training Args")
    ### Environment params
    parser.add_argument(
        "--env-name",
        "-env",
        default="GridWorld-Default",
        help="gym environment to use (default: GridWorld-Default)",
    )

    parser.add_argument(
        "--puzzle-path",
        default="maps/test_v4.txt",
        help="gridworld textfile to use (default: maps/test_v4.txt)",
    )

    parser.add_argument("--wandb", action="store_true", help="whether to use wandb logging (default: False)")

    parser.add_argument("--sparse", action="store_true", help="make environment sparse (default:False)")

    ## Training params
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed (default: 0). 0 means no seeding",
    )

    parser.add_argument(
        "--method",
        type=str,
        default="ppo",
        choices=[
            "ppo",
            "ppo-hca",
        ],  
        help="Method we are running: one of ppo or ppo_ca (default: ppo)",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="device to run on (default: cpu)"
    )
    parser.add_argument(
        "--max-training-episodes",
        type=int,
        default=500000,
        help="maxmimum training episodes (default: 500000)",
    )
    parser.add_argument(
        "--update-every",
        type=int,
        default=50,
        help="update policy every these many episodes (default: 50)",
    )
    parser.add_argument(
        "--ppo-epochs",
        type=int,
        default=30,
        help="update policy for K epochs in one PPO update (default:30)",
    )

    parser.add_argument("--dont-update", action="store_true", help="dont update models (default:False)")

    parser.add_argument(
        "--eps-clip",
        type=float,
        default=0.2,
        help="clip parameter for PPO (default: 0.2)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="discount factor (default:0.99)",
    )
    parser.add_argument(
        "--lamda", type=float, default=0.95, help="GAE lambda (default:0.95)"
    )
    parser.add_argument(
        "--entropy-coeff",
        type=float,
        default=0.0,
        help="Entropy Coefficient (default:0.0)",
    )
    parser.add_argument(
        "--value-loss-coeff",
        type=float,
        default=0.25,
        help="Value Loss Coefficient (default:0.25)",
    )
    parser.add_argument(
        "--adv",
        type=str,
        default="gae",
        choices=["gae", "mc", "hca"],
        help="What advantage calculation to use (default: gae)",
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=2,
        help="number of hidden layers (default: 2)",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=64,
        help="hidden size of models (default:64)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="learning rate for actor network (default: 3e-4)",
    )

    ## Saving and logging:
    parser.add_argument(
        "--log-freq",
        type=int,
        default=2500,
        help="Log frequency in episodes. Use 0 for no logging (default:2500)",
    )
    parser.add_argument(
        "--save-model-freq",
        type=int,
        default=1000000,
        help="Model save frequency in episodes. Use 0 for no saving (default: 1000000)",
    )

    parser.add_argument(
        "--collect-hca-data",
        action="store_true",
        help="Whether to store transition data for HCA function training.",
    )

    parser.add_argument(
        "--hca-data-save-freq",
        type=int,
        default=5000,
        help="How many episodes between HCA data getting saved (default:5000)"
    )

    parser.add_argument(
        "--hca-checkpoint",
        type=str,
        default="checkpoints/hca:test_v4_75000_eps_2023-01-31 17:27:42/model_99.pt",
        help="path to HCA model checkpoint"
    )

    parser.add_argument("--eval-freq", type=int, default=10000, help="How often to run evaluation on agent.")

    ## Loading checkpoints:
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=str,
        default="",
        help="path to checkpoint (default: "
        "). Empty string does not load a checkpoint.",
    )

    args = parser.parse_args()

    # sanity check
    if args.method == 'ppo-hca':
        args.adv = 'hca'
        args.value_loss_coeff = 0.
        print("Using method PPO-HCA: Setting the advantage calculation to hca and value-loss coefficient to 0")
    train(args)
    