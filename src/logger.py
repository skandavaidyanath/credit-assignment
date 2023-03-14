import dataclasses
from dataclasses import dataclass, asdict

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
class Stats:
    """Class for training stats logging"""

    avg_rewards: float
    avg_success: float
    total_loss: float
    action_loss: float
    value_loss: float
    entropy: float
    hca_ratio_mean: float = 0.0
    hca_ratio_std: float = 0.0
    hca_ratio_min: float = 0.0
    hca_ratio_max: float = 0.0
    hca_train_loss: float = 0.0
    hca_train_logprobs: float = 0.0
    hca_train_acc: float = 0.0
    hca_val_loss: float = 0.0
    hca_val_logprobs: float = 0.0
    hca_val_acc: float = 0.0


class Logger:
    """
    Class for logging stats
    """

    def __init__(self, exp_name, env_name, config, entity, wandb=False):
        self.wandb = wandb

        if self.wandb:
            wandb.init(
                name=exp_name,
                project=env_name,
                config=config,
                entity=entity,
            )

    def log(self, stats, step, wandb_prefix="training"):
        if self.wandb:
            self.wandb_log(stats, step, wandb_prefix)
        else:
            if dataclasses.is_dataclass(stats):
                stats_dict = asdict(stats)
            else:
                stats_dict = stats
            print(
                f"Steps: {step} \t\t Average Reward: {stats_dict['avg_rewards']:.4f} \t\t Average Success: {stats_dict['avg_success']:.4f}"
            )

    def wandb_log(self, stats, step, wandb_prefix):
        if isinstance(stats, dataclass):
            stats_dict = asdict(stats)
        else:
            stats_dict = stats

        prefixed_stats_dict = self.prefix(stats_dict, wandb_prefix)

        wandb.log(prefixed_stats_dict, step)

    def prefix(self, d, pre):
        new_d = {}
        for k, v in d.items():
            new_d[pre + "/" + k] = v
        return new_d
