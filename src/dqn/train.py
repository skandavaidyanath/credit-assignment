import argparse
import datetime
import os
import gym
import numpy as np
import torch
import wandb

from gridworld import GridWorld
from eval import eval
from dqn.dqn_algo import DQN
from dqn.replay_buffer import ReplayBuffer


def train(args):
    # Environment
    env = GridWorld(args.puzzle_path, sparse=args.sparse)

    if isinstance(env.action_space, gym.spaces.Box):
        continuous = True
    else:
        continuous = False

    if continuous:
        raise Exception("DQN does not support continuous action spaces!")
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
    if args.cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    checkpoint_path = None
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
            entity="svaidyan",
        )

    # Agent
    agent = DQN(state_dim, action_dim, args.lr, device, args.eps_decay_steps, eps_init=args.eps_init, eps_fin=args.eps_fin,
                gamma=args.gamma, n_layers=args.n_layers, hidden_size=args.hidden_size, max_grad_norm=args.max_grad_norm,
                loss_fn=args.loss_fn)


    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        agent.load(checkpoint)

    buffer = ReplayBuffer(state_dim, act_dim=1, max_buffer_size=args.max_buffer_size)

    total_rewards, total_successes, total_losses = [], [], []

    # track total training time
    start_time = datetime.datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print(
        "============================================================================================"
    )

    num_env_steps = 0
    num_train_steps = 0


    for episode in range(1, args.max_training_episodes + 1):
        state = env.reset()
        current_ep_reward = 0
        done = False
        info = None

        warmup_episode = (episode < args.num_warmup_episodes) or (num_env_steps < args.batch_size)

        while not done:
            # update the policy
            if not warmup_episode and num_env_steps % args.update_policy_freq == 0:
                losses = []
                for _ in range(args.num_steps_per_update):
                    batch = buffer.sample(args.batch_size)
                    loss = agent.train_step(batch)
                    losses.append(loss.item())
                    num_train_steps += 1

                    # Periodically update the target network by Q network to target Q network
                    if num_train_steps % args.target_q_update_rate == 0:
                        agent.target_q_network.load_state_dict(agent.q_net.state_dict())

                total_losses.append(np.mean(np.array(losses)))

            # get action with policy
            action = agent.select_action(state, random=warmup_episode, t=num_env_steps)
            action = action.item()

            # step the environment and add transition to buffer
            next_state, reward, done, info = env.step(action)
            buffer.add_transition(state, action, reward, next_state, done)

            current_ep_reward += reward
            state = next_state
            num_env_steps += 1

        total_rewards.append(current_ep_reward)
        total_successes.append(info["success"])

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
                        "training/train_steps": num_train_steps,
                        "training/env_steps": num_env_steps,
                    },
                    step=episode,
                )

            print(
                f"Episode: {episode} \t\t Average Reward: {avg_reward:.4f} \t\t Average Success: {avg_success:.4f}"
            )

            total_rewards, total_successes, total_losses = [], [], []

        # save model weights
        if args.save_model_freq and episode % args.save_model_freq == 0:
            assert checkpoint_path is not None
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DQN Training Args")
    ### Environment params
    parser.add_argument(
        "--env-name",
        "-env",
        default="GridWorld-Default",
        help="gym environment to use (default: GridWorld-Default)",
    )

    parser.add_argument(
        "--puzzle-path",
        default="maps/test_v1.txt",
        help="gridworld textfile to use (default: maps/test_v1.txt)",
    )

    parser.add_argument("--wandb", type=bool, default=False, help="whether to use wandb logging (default: False)")

    parser.add_argument("--sparse", type=bool, default=False, help="make environment sparse (default:False)")

    ## Training params
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed (default: 0). 0 means no seeding",
    )

    parser.add_argument("--eval_freq", type=int, default=100, help="How often to run evaluation on agent.")

    parser.add_argument(
        "--method",
        type=str,
        default="dqn",
        choices=[
            "dqn",
            "dqn-ca",
        ],
        help="Method we are running: one of dqn or dqn-ca (default: dqn)",
    )
    parser.add_argument(
        "--cuda", type=bool, default=False, help="run on CUDA (default: False)"
    )
    parser.add_argument(
        "--max-training-episodes",
        type=int,
        default=500000,
        help="maxmimum training episodes (default: 500000)",
    )
    parser.add_argument(
        "--update-policy-freq",
        type=int,
        default=1,
        help="update policy every these many env steps (default: 1)",
    )
    parser.add_argument(
        "--num-steps-per-update",
        type=int,
        default=1,
        help="perform these many gradient updates per policy update (default: 1)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="training batch size for DQN (default:32)",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="discount factor (default:0.99)",
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
        help="learning rate for dqn network (default: 3e-4)",
    )

    parser.add_argument(
        "--eps-init",
        type=float,
        default=1.0,
        help="Initial value for linear decayed epsilon-greedy exploration."
    )

    parser.add_argument(
        "--eps-fin",
        type=float,
        default=0.1,
        help="Final value for linear decayed epsilon-greedy exploration."
    )

    parser.add_argument(
        "--eps-decay-steps",
        type=float,
        default=100000,
        help="Number of env steps for epsilon to decay to final value."
    )

    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=10.0,
        help="Maximum grad norm when performing gradient updates."
    )

    parser.add_argument(
        "--loss-fn",
        type=str,
        default="huber",
        help="DQN loss function to use."
    )

    parser.add_argument(
        "--max-buffer-size",
        type=int,
        default=1000000,
        help="Maximum grad norm when performing gradient updates."
    )

    parser.add_argument(
        "--target-q-update_rate",
        type=int,
        default=10000,
        help="Number of gradient steps between updates to the target net"
    )

    parser.add_argument(
        "--num-warmup-episodes",
        type=int,
        default=300,
        help="Number of episodes in which the agent acts purely randomly (used to init buffer)."
    )

    ## Saving model:
    parser.add_argument(
        "--log-freq",
        type=int,
        default=300,
        help="Log frequency in episodes. Use 0 for no logging (default:2500)",
    )
    parser.add_argument(
        "--save-model-freq",
        type=int,
        default=1000000,
        help="Model save frequency in episodes. Use 0 for no saving (default: 1000000)",
    )

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
    train(args)
