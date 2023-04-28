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

from ppo.model import PPO
from ppo.buffer import RolloutBuffer

from hca.model import HCAModel
from hca.buffer import HCABuffer, calculate_mc_returns

from dualdice.dd_model import DualDICE
from dualdice.dd_buffer import DualDICEBuffer
from dualdice.return_model import ReturnPredictor
from dualdice.return_buffer import ReturnBuffer

from utils import (
    assign_hindsight_info,
    get_hindsight_actions,
    get_env,
)
from eval import eval
from logger import PPO_Stats, HCA_Stats, DD_Stats, Return_Stats, Logger


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
    h_model, hca_buffer = None, None
    if args.agent.name in ["ppo-hca", "hca-dualdice"]:
        h_model = HCAModel(
            state_dim + 1,  # +1 is for return-conditioned
            action_dim,
            continuous=continuous,
            n_layers=args.agent.hca_n_layers,
            hidden_size=args.agent.hca_hidden_size,
            activation_fn=args.agent.hca_activation,
            dropout_p=args.agent.hca_dropout,
            batch_size=args.agent.hca_batchsize,
            lr=args.agent.hca_lr,
            device=args.training.device,
            normalize_inputs=args.agent.hca_normalize_inputs,
            max_grad_norm=args.agent.hca_max_grad_norm,
            weight_training_samples=args.agent.hca_weight_training_samples,
            noise_std=args.agent.hca_noise_std,
        )
        h_model = h_model.to(args.training.device)
        if args.agent.hca_checkpoint:
            hca_checkpoint = torch.load(args.agent.hca_checkpoint)
            h_model.load(hca_checkpoint["model"], strict=True)
            print(
                f"Successfully loaded hca model from {args.agent.hca_checkpoint}!"
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

    # DualDICE model
    dd_model, dd_buffer = None, None
    if args.agent.name in ["hca-dualdice"]:
        dd_act_dim = action_dim if continuous else 1

        dd_model = DualDICE(
            state_dim=state_dim,
            action_dim=dd_act_dim,
            f=args.agent.dd_f,
            n_layers=args.agent.hca_n_layers,
            hidden_size=args.agent.hca_hidden_size,
            activation_fn=args.agent.hca_activation,
            dropout_p=args.agent.hca_dropout,
            batch_size=args.agent.hca_batchsize,
            lr=args.agent.hca_lr,
            device=args.training.device,
            normalize_inputs=args.agent.hca_normalize_inputs,
            max_grad_norm=args.agent.dd_max_grad_norm
        )

        dd_buffer = DualDICEBuffer(
            action_dim=dd_act_dim,
            train_val_split=args.agent.hca_train_val_split,
        )

        r_model = ReturnPredictor(
            state_dim=state_dim,
            quantize=args.agent.r_quant,
            num_classes=args.agent.r_num_classes,
            n_layers=args.agent.hca_n_layers,
            hidden_size=args.agent.hca_hidden_size,
            activation_fn=args.agent.hca_activation,
            dropout_p=args.agent.hca_dropout,
            batch_size=args.agent.hca_batchsize,
            lr=args.agent.hca_lr,
            device=args.training.device,
            normalize_inputs=args.agent.hca_normalize_inputs,
            normalize_targets=args.agent.r_normalize_targets,
            max_grad_norm=args.agent.r_max_grad_norm
        )

        r_buffer = ReturnBuffer(
            num_classes=args.agent.r_num_classes if args.agent.r_quant else 1,
            train_val_split=args.agent.hca_train_val_split,
        )

    # Replay Buffer for PPO
    buffer = RolloutBuffer()

    # logging
    total_rewards, total_successes = [], []
    ep_lens = []
    total_losses, action_losses, value_losses, entropies = [], [], [], []

    ca_stat_mins, ca_stat_maxes, ca_stat_means, ca_stat_stds = (
        [],
        [],
        [],
        [],
    )
    ca_stat_type = ""
    env_steps_between_policy_updates = 0
    env_steps_between_ca_updates = 0
    num_policy_updates = 0

    # track total training time
    start_time = datetime.datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print(
        "============================================================================================"
    )

    # initial eval
    print(" ============ Evaluating =============")
    eval_avg_reward, eval_avg_success, eval_avg_ep_len = eval(env, agent, args)
    logger.log(
        {
            "avg_rewards": eval_avg_reward,
            "avg_success": eval_avg_success,
            "avg_ep_len": eval_avg_ep_len,
        },
        step=0,
        wandb_prefix="eval",
    )
    print("======= Finished Evaluating =========")

    for episode in range(1, args.training.max_training_episodes + 1):
        state = env.reset()

        current_ep_reward = 0
        done = False

        ep_len = 0

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
            ep_len += 1

        total_rewards.append(current_ep_reward)
        if "success" in info:
            total_successes.append(info["success"])
        else:
            total_successes.append(0.0)

        if args.agent.name in ["ppo-hca", "hca-dualdice"]:
            returns = calculate_mc_returns(rewards, terminals, agent.gamma)
            buffer.returns.append(returns)

        buffer.states.append(states)
        buffer.actions.append(actions)
        buffer.logprobs.append(logprobs)
        buffer.rewards.append(rewards)
        buffer.terminals.append(terminals)

        env_steps_between_policy_updates += ep_len
        env_steps_between_ca_updates += ep_len
        ep_lens.append(ep_len)

        # Add data for HCA (and DualDice).
        if args.agent.name in ["ppo-hca", "hca-dualdice"]:
            hca_buffer.add_episode(states, actions, rewards, agent.gamma)

            if args.agent.name in ["hca-dualdice"]:
                h_actions = get_hindsight_actions(h_model, states, returns)
                pi_actions = actions
                dd_buffer.add_episode(states, h_actions, pi_actions, returns)

                r_buffer.add_episode(states, returns)

        # Determine whether the policy will be updated now or not.
        if args.agent.get("update_every_env_steps"):
            time_for_policy_update = (
                env_steps_between_policy_updates
                >= args.agent.update_every_env_steps
            )
        else:
            time_for_policy_update = episode % args.agent.update_every == 0
            
        # Determine whether the hindsight functions will be updated now or not.
        if args.agent.name in ["ppo-hca", "hca-dualdice"]:
            if args.agent.get("hca_update_every_env_steps"):
                time_for_ca_update = (
                    env_steps_between_ca_updates
                    >= args.agent.hca_update_every_env_steps
                )
            else:
                time_for_ca_update = episode % args.agent.hca_update_every == 0
            
        

        # Update credit assignment (hca) model, if needed.
        # Always update the HCA model the first time before a PPO update.
        first_policy_update = time_for_policy_update and num_policy_updates == 0
        if args.agent.name in ["ppo-hca", "hca-dualdice"] and (
            time_for_ca_update or first_policy_update
        ):

            # normalize inputs if required
            if h_model.normalize_inputs:
                input_mean, input_std = hca_buffer.get_input_stats()
                h_model.update_norm_stats(
                    input_mean, input_std, args.agent.refresh_hca
                )

            # reset the model if you want
            if args.agent.refresh_hca:
                h_model.reset_parameters()

            # update the HCA model
            for _ in range(args.agent.hca_epochs):
                hca_results = h_model.update(hca_buffer)

            # Clear the HCA buffer
            hca_buffer.clear()

            if hca_results:
                # Log every time we update the model and don't use the log freq
                hca_stats = HCA_Stats(**hca_results)

                print(" ============ Updated HCA model =============")
                logger.log(hca_stats, step=episode, wandb_prefix="training")
                print("=============================================")

            if args.agent.name in ["hca-dualdice"]:
                # normalize inputs if required
                if dd_model.normalize_inputs:
                    h_mean, h_std, pi_mean, pi_std = dd_buffer.get_input_stats()
                    dd_model.update_norm_stats(
                        h_mean, h_std, pi_mean, pi_std, args.agent.refresh_hca
                    )

                # reset the model if you want
                if args.agent.refresh_hca:
                    dd_model.reset_parameters()

                # update the DD model
                # using a separate argument dd_epochs here
                for _ in range(args.agent.dd_epochs):
                    dd_results = dd_model.update(dd_buffer)

                # Clear the DD buffer
                dd_buffer.clear()

                # Log every time we update the model and don't use the log freq
                dd_stats = DD_Stats(**dd_results)

                print(" ============ Updated DD model =============")
                logger.log(dd_stats, step=episode, wandb_prefix="training")
                print("=============================================")

                # Return model update
                # normalize inputs if required
                if r_model.normalize_inputs:
                    input_mean, input_std = r_buffer.get_input_stats()
                    r_model.update_norm_stats(
                        input_mean, input_std, args.agent.refresh_hca
                    )
                if r_model.normalize_targets:
                    target_mean, target_std = r_buffer.get_target_stats()
                    r_model.update_target_stats(
                        target_mean, target_std, args.agent.refresh_hca
                    )

                # reset the model if you want
                if args.agent.refresh_hca:
                    r_model.reset_parameters()

                # update the Return model
                # using a separate argument r_epochs here
                for _ in range(args.agent.r_epochs):
                    ret_results = r_model.update(r_buffer)

                # Clear the Return buffer
                r_buffer.clear()

                # Log every time we update the model and don't use the log freq
                ret_stats = Return_Stats(**ret_results)

                print(" ============ Updated Return model =============")
                logger.log(ret_stats, step=episode, wandb_prefix="training")
                print("=============================================")
                
            env_steps_between_ca_updates = 0

        # Agent update (PPO)
        if args.agent.name != "random" and time_for_policy_update:
            if args.agent.name in ["ppo-hca"]:
                # First, assign credit to the actions in the data.
                assign_hindsight_info(buffer, h_model=h_model)
            elif args.agent.name in ["hca-dualdice"]:
                # Assign the density ratios directly using DD model
                # and return model
                # Product of DD model and R model will give the \pi/h ratio
                assign_hindsight_info(
                    buffer,
                    dd_model=dd_model,
                    r_model=r_model,
                )

            # Perform the actual PPO update.
            (
                total_loss,
                action_loss,
                value_loss,
                entropy,
                ca_stat_dict,
            ) = agent.update(buffer)
            total_losses.append(total_loss)
            action_losses.append(action_loss)
            value_losses.append(value_loss)
            entropies.append(entropy)
            if ca_stat_dict:
                ca_stat_type = ca_stat_dict["ca_stat_type"]
                ca_stat_mins.append(ca_stat_dict["min"])
                ca_stat_maxes.append(ca_stat_dict["max"])
                ca_stat_means.append(ca_stat_dict["mean"])
                ca_stat_stds.append(ca_stat_dict["std"])

            buffer.clear()
            env_steps_between_policy_updates = 0
            num_policy_updates += 1

        # logging
        if args.training.log_freq and episode % args.training.log_freq == 0:

            avg_reward = np.mean(total_rewards)
            avg_success = np.mean(total_successes)
            avg_ep_len = np.mean(ep_lens)

            ca_stat_min = (
                np.mean(ca_stat_mins) if len(ca_stat_mins) > 0 else None
            )
            ca_stat_max = (
                np.mean(ca_stat_maxes) if len(ca_stat_maxes) > 0 else None
            )
            ca_stat_mean = (
                np.mean(ca_stat_means) if len(ca_stat_means) > 0 else None
            )
            ca_stat_std = (
                np.mean(ca_stat_stds) if len(ca_stat_stds) > 0 else None
            )

            total_loss = None if len(total_losses)==0 else np.mean(total_losses)
            action_loss = (
                None if len(action_losses)==0 else np.mean(action_losses)
            )
            value_loss = None if len(value_losses)==0 else np.mean(value_losses)
            entropy = None if len(entropies)==0 else np.mean(entropies)

            stats = PPO_Stats(
                avg_rewards=avg_reward,
                avg_success=avg_success,
                avg_ep_len=avg_ep_len,
                total_loss=total_loss,
                action_loss=action_loss,
                value_loss=value_loss,
                entropy=entropy,
                ca_stat=ca_stat_type,
                ca_stat_mean=ca_stat_mean,
                ca_stat_std=ca_stat_std,
                ca_stat_max=ca_stat_max,
                ca_stat_min=ca_stat_min,
            )

            logger.log(stats, step=episode, wandb_prefix="training")

            total_rewards, total_successes = [], []
            ep_lens = []
            total_losses, action_losses, value_losses, entropies = (
                [],
                [],
                [],
                [],
            )

            ca_stat_mins, ca_stat_maxes, ca_stat_means, ca_stat_stds = (
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
            eval_avg_reward, eval_avg_success, eval_avg_ep_len = eval(
                env, agent, args
            )

            print(" ============ Evaluating =============")
            logger.log(
                {
                    "avg_rewards": eval_avg_reward,
                    "avg_success": eval_avg_success,
                    "avg_ep_len": eval_avg_ep_len,
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
