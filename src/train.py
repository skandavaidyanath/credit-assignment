import hydra
from omegaconf import DictConfig, OmegaConf
import datetime
import os
import random
import warnings

# suppress D4RL warnings
os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"

# suppress training split warnings in auxiliary models training
warnings.filterwarnings(
    "ignore",
    "Length of split at index 1 is 0. This might result in an empty dataset",
)


import gym
import numpy as np
import torch

from ppo.model import PPO
from ppo.buffer import RolloutBuffer

from arch.cnn import CNNBase

from hca.model import HCAModel
from hca.buffer import HCABuffer, calculate_mc_returns

from dualdice.dd_model import DualDICE
from dualdice.dd_buffer import DualDICEBuffer
from dualdice.return_model import ReturnPredictor
from dualdice.return_buffer import ReturnBuffer

from utils import (
    assign_hindsight_info,
    get_hindsight_actions,
    get_psi_return_samples,
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

    if args.env.type == "gridworld":
        # gridworld env
        input_dim = env.observation_space["map"].shape[0] + 1
    elif args.env.type == "atari":
        # Atari env
        state = env.reset()
        input_dim = state.shape[0]
        image_dim = state.shape
    else:
        input_dim = env.observation_space.shape[0]

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

    reward_type = "delayed" if args.env.delay_reward else "dense"
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

    # Need to train the Value network if we plan to stop HCA at some point
    if args.agent.stop_hca:
        assert (
            args.agent.value_loss_coeff >= 0.0
        ), "Please provide a value loss coefficient >=0 when stopping HCA!"

    # Agent
    if args.env.type == "atari":
        ppo_cnn = CNNBase(
            num_inputs=input_dim, hidden_size=args.agent.hidden_size
        )
    else:
        ppo_cnn = None
    agent = PPO(
        input_dim, action_dim, args.agent.lr, continuous, device, args, ppo_cnn
    )

    if args.training.checkpoint:
        checkpoint = torch.load(args.training.checkpoint)
        agent.load(checkpoint["policy"])

    # HCA model
    h_model, hca_buffer = None, None
    if args.agent.name in ["ppo-hca", "hca-dualdice"]:
        if args.env.type == "atari":
            hca_cnn = CNNBase(
                num_inputs=input_dim, hidden_size=args.agent.hca_hidden_size
            )
        else:
            hca_cnn = None

        h_model = HCAModel(
            args.agent.hca_hidden_size + 1
            if args.env.type == "atari"
            else input_dim + 1,  # +1 is for return-conditioned
            action_dim,
            continuous=continuous,
            cnn_base=hca_cnn,
            n_layers=args.agent.hca_n_layers,
            hidden_size=args.agent.hca_hidden_size,
            activation_fn=args.agent.hca_activation,
            dropout_p=args.agent.hca_dropout,
            batch_size=args.agent.hca_batchsize,
            lr=args.agent.hca_lr,
            device=args.training.device,
            normalize_inputs=args.agent.hca_normalize_inputs,
            normalize_return_inputs_only=args.agent.hca_normalize_return_inputs_only,
            max_grad_norm=args.agent.hca_max_grad_norm,
            weight_training_samples=args.agent.hca_weight_training_samples,
            noise_std=args.agent.hca_noise_std,
        )
        if args.agent.hca_checkpoint:
            hca_checkpoint = torch.load(args.agent.hca_checkpoint)
            h_model.load(hca_checkpoint["model"], strict=True)
            print(
                f"Successfully loaded hca model from {args.agent.hca_checkpoint}!"
            )

        # HCA Buffer
        if continuous:
            hca_buffer = HCABuffer(
                state_dim=image_dim if args.env.type == "atari" else input_dim,
                action_dim=action_dim,
                train_val_split=args.agent.hca_train_val_split,
            )
        else:
            hca_buffer = HCABuffer(
                state_dim=image_dim if args.env.type == "atari" else input_dim,
                action_dim=1,
                train_val_split=args.agent.hca_train_val_split,
            )

    # DualDICE model
    dd_model, dd_buffer = None, None
    dd_cnn, r_cnn = None, None
    if args.agent.name in ["hca-dualdice"]:
        dd_act_dim = action_dim if continuous else 1

        if args.env.type == "atari":
            dd_cnn = CNNBase(
                num_inputs=input_dim, hidden_size=args.agent.hca_hidden_size
            )
            r_cnn = CNNBase(
                num_inputs=input_dim, hidden_size=args.agent.hca_hidden_size
            )

        dd_model = DualDICE(
            state_dim=args.agent.hca_hidden_size
            if args.env.type == "atari"
            else input_dim,
            action_dim=dd_act_dim,
            cnn_base=dd_cnn,  # using different CNNs here not worried about compute
            f=args.agent.dd_f,
            n_layers=args.agent.hca_n_layers,
            hidden_size=args.agent.hca_hidden_size,
            activation_fn=args.agent.hca_activation,
            dropout_p=args.agent.hca_dropout,
            batch_size=args.agent.hca_batchsize,
            lr=args.agent.hca_lr,
            device=args.training.device,
            normalize_inputs=args.agent.hca_normalize_inputs,
            normalize_return_inputs_only=args.agent.hca_normalize_return_inputs_only,
            max_grad_norm=args.agent.dd_max_grad_norm,
        )

        dd_buffer = DualDICEBuffer(
            state_dim=image_dim if args.env.type == "atari" else input_dim,
            action_dim=dd_act_dim,
            train_val_split=args.agent.hca_train_val_split,
        )

        r_model = ReturnPredictor(
            state_dim=args.agent.hca_hidden_size
            if args.env.type == "atari"
            else input_dim,
            quantize=args.agent.r_quant,
            num_classes=args.agent.r_num_classes,
            cnn_base=r_cnn,  # using different CNNs here not worried about compute
            n_layers=args.agent.hca_n_layers,
            hidden_size=args.agent.hca_hidden_size,
            activation_fn=args.agent.hca_activation,
            dropout_p=args.agent.hca_dropout,
            batch_size=args.agent.hca_batchsize,
            lr=args.agent.hca_lr,
            device=args.training.device,
            normalize_inputs=args.agent.hca_normalize_inputs,
            normalize_targets=args.agent.r_normalize_targets,
            max_grad_norm=args.agent.r_max_grad_norm,
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
                pi_actions = actions
                dd_buffer.add_episode(states, pi_actions, returns)
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
                (
                    h_state_mean,
                    h_state_std,
                    h_return_mean,
                    h_return_std,
                ) = hca_buffer.get_input_stats(
                    h_model.normalize_return_inputs_only
                )
                h_model.update_norm_stats(
                    h_state_mean,
                    h_state_std,
                    h_return_mean,
                    h_return_std,
                    args.agent.refresh_hca,
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

                # print(" ============ Updated HCA model =============")
                logger.log(hca_stats, step=episode, wandb_prefix="training")
                # print("=============================================")

            if args.agent.name in ["hca-dualdice"]:
                # Update return model first, as this is needed for the dualdice model update.
                # normalize inputs if required
                if r_model.normalize_inputs:
                    r_inp_mean, r_inp_std = r_buffer.get_input_stats()
                    r_model.update_norm_stats(
                        r_inp_mean, r_inp_std, args.agent.refresh_hca
                    )
                if r_model.normalize_targets:
                    r_target_mean, r_target_std = r_buffer.get_target_stats()
                    r_model.update_target_stats(
                        r_target_mean, r_target_std, args.agent.refresh_hca
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

                # print(" ============ Updated Return model =============")
                logger.log(ret_stats, step=episode, wandb_prefix="training")
                # print("=============================================")

                # compute the hindsight actions for the dualdice buffer
                h_actions = get_hindsight_actions(
                    h_model, dd_buffer.states, dd_buffer.returns
                )
                dd_buffer.h_actions.extend(h_actions)

                # Compute the return samples used to estimate the second expectation in the DualDICE Loss.
                r_min, r_max = (
                    np.array(dd_buffer.returns).min(),
                    np.array(dd_buffer.returns).max(),
                )
                r_samples = get_psi_return_samples(
                    args.agent.psi, r_model, dd_buffer.states, r_min, r_max
                )
                dd_buffer.psi_returns.extend(r_samples)

                # normalize inputs if required
                if dd_model.normalize_inputs:
                    (
                        dd_state_mean,
                        dd_state_std,
                        dd_action_mean,
                        dd_action_std,
                        dd_return_mean,
                        dd_return_std,
                    ) = dd_buffer.get_input_stats(
                        dd_model.normalize_return_inputs_only
                    )
                    dd_model.update_norm_stats(
                        dd_state_mean,
                        dd_state_std,
                        dd_action_mean,
                        dd_action_std,
                        dd_return_mean,
                        dd_return_std,
                        args.agent.refresh_hca,
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

                # print(" ============ Updated DD model =============")
                logger.log(dd_stats, step=episode, wandb_prefix="training")
                # print("=============================================")

            env_steps_between_ca_updates = 0

        # Agent update (PPO)
        if args.agent.name != "random" and time_for_policy_update:
            if args.agent.name in ["ppo-hca"]:
                # First, assign credit to the actions in the data.
                assign_hindsight_info(buffer, h_model=h_model)
            elif args.agent.name in ["hca-dualdice"]:
                # Assign the density ratios using the DD model and (potentially) the return model

                # If returns are sampled in the dd_loss using the r_model, then the DD model output does NOT need to be
                # multiplied with the return model. If not, then the hindsight ratio is given by the product of the
                # DD model and the return model.
                take_r_product = args.agent.psi != "r_model"
                assign_hindsight_info(
                    buffer,
                    dd_model=dd_model,
                    r_model=r_model,
                    take_r_product=take_r_product,
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

            # At the end of this update, switch from HCA algo to regular PPO if required
            if (
                args.agent.name != "ppo"
                and args.agent.stop_hca
                and episode >= args.agent.stop_hca
            ):
                print(
                    "##################################################################"
                )
                print("STOPPING HCA, SWITCHING TO VANILLA PPO")
                print(
                    "##################################################################"
                )
                args.agent.name = "ppo"
                agent.adv = "gae"

        # logging
        if args.training.log_freq and episode % args.training.log_freq == 0:
            avg_reward = np.mean(total_rewards)
            avg_success = np.mean(total_successes)
            avg_ep_len = np.mean(ep_lens)

            ca_stat_min = (
                np.mean(ca_stat_mins) if len(ca_stat_mins) > 0 else np.nan
            )
            ca_stat_max = (
                np.mean(ca_stat_maxes) if len(ca_stat_maxes) > 0 else np.nan
            )
            ca_stat_mean = (
                np.mean(ca_stat_means) if len(ca_stat_means) > 0 else np.nan
            )
            ca_stat_std = (
                np.mean(ca_stat_stds) if len(ca_stat_stds) > 0 else np.nan
            )

            total_loss = (
                np.nan if len(total_losses) == 0 else np.mean(total_losses)
            )
            action_loss = (
                np.nan if len(action_losses) == 0 else np.mean(action_losses)
            )
            value_loss = (
                np.nan if len(value_losses) == 0 else np.mean(value_losses)
            )
            entropy = np.nan if len(entropies) == 0 else np.mean(entropies)

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
