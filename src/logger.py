from dataclasses import dataclass, asdict, is_dataclass

import numpy as np
import wandb


def stat(x, stat="mean"):
    """
    Calculates the statistic with numpy and returns 0. if x is empty/ None
    Default stat is mean
    """
    stat_func = getattr(np, stat)
    if x:
        return stat_func(x)
    else:
        return 0.0


@dataclass
class PPO_Stats:
    """Class for PPO training stats logging"""

    avg_rewards: float
    avg_success: float
    avg_ep_len: float
    total_loss: float
    action_loss: float
    value_loss: float
    entropy: float

    ca_stat: str = "ca_stat"
    ca_stat_mean: float = 0.0
    ca_stat_std: float = 0.0
    ca_stat_min: float = 0.0
    ca_stat_max: float = 0.0


@dataclass
class HCA_Stats:
    """Class for HCA training stats logging"""

    hca_train_loss: float = 0.0
    hca_train_logprobs: float = 0.0
    hca_train_acc: float = 0.0
    hca_val_loss: float = 0.0
    hca_val_logprobs: float = 0.0
    hca_val_acc: float = 0.0

    hca_train_entropy_min: float = 0.0
    hca_train_entropy_max: float = 0.0
    hca_train_entropy_mean: float = 0.0
    hca_train_entropy_std: float = 0.0


@dataclass
class DD_Stats:
    """Class for DualDICE training stats logging"""

    dd_train_loss: float = 0.0
    dd_val_loss: float = 0.0


@dataclass
class Return_Stats:
    """Class for Return predictor training stats logging"""

    ret_train_loss: float = 0.0
    ret_train_logprobs: float = 0.0
    ret_train_acc: float = 0.0
    ret_val_loss: float = 0.0
    ret_val_logprobs: float = 0.0
    ret_val_acc: float = 0.0


class Logger:
    """
    Class for logging stats
    """

    def __init__(
        self,
        exp_name,
        env_name,
        agent_name,
        config,
        entity,
        use_wandb=False,
        group_modifier="",
    ):
        self.use_wandb = use_wandb
        group_name = (
            agent_name + "_" + group_modifier if group_modifier else agent_name
        )
        if self.use_wandb:
            wandb.init(
                name=exp_name,
                project=env_name,
                group=group_name,
                config=config,
                entity=entity,
            )

    def log(self, stats, step, wandb_prefix="training"):
        if self.use_wandb:
            self.wandb_log(stats, step, wandb_prefix)
        if isinstance(stats, PPO_Stats):
            print(
                f"Episode: {step} \t\t Average Reward: {stats.avg_rewards:.4f} \t\t Average Success: {stats.avg_success:.4f}"
            )
        elif isinstance(stats, HCA_Stats):
            # print(
            #     f"HCA Train Stats --- Loss: {stats.hca_train_loss} | Logprobs: {stats.hca_train_logprobs} | Acc: {stats.hca_train_acc}"
            # )
            # print(
            #     f"HCA Val Stats --- Loss: {stats.hca_val_loss} | Logprobs: {stats.hca_val_logprobs} | Acc: {stats.hca_val_acc}"
            # )
            pass
        elif isinstance(stats, DD_Stats):
            # print(f"DualDice Train Loss: {stats.dd_train_loss}")
            # print(f"DualDice Val Loss: {stats.dd_val_loss}")
            pass
        elif isinstance(stats, Return_Stats):
            # print(
            #     f"Return Model Train Stats --- Loss: {stats.ret_train_loss} | Logprobs: {stats.ret_train_logprobs} | Acc: {stats.ret_train_acc}"
            # )
            # print(
            #     f"Return model Val Stats --- Loss: {stats.ret_val_loss} | Logprobs: {stats.ret_val_logprobs} | Acc: {stats.ret_val_logprobs}"
            # )
            pass
        else:
            # stats is a dictionary during eval
            print("\t Episode: ", step)
            print("\t Average eval returns: ", stats["avg_rewards"])
            print("\t Average eval success: ", stats["avg_success"])

    def wandb_log(self, stats, step, wandb_prefix):
        if is_dataclass(stats):
            stats_dict = asdict(stats)
        else:
            stats_dict = stats

        stats_dict = self.remove_nans(stats_dict)

        prefixed_stats_dict = self.prefix(stats_dict, wandb_prefix)

        wandb.log(prefixed_stats_dict, step)

    def remove_nans(self, d):
        """
        Remove nans that arise from inconcsistent logging schedules
        before logging
        """
        new_d = {k: v for k, v in d.items() if v is not np.nan}
        return new_d

    def prefix(self, d, pre):
        ca_stat_type = d.pop("ca_stat", "")
        new_d = {}
        for k, v in d.items():
            if "ca_stat" in k and ca_stat_type:
                k = k.replace("ca_stat", ca_stat_type)
            new_d[pre + "/" + k] = v
        return new_d
