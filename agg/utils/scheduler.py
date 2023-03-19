"""Copyright 2023 by @jet981217. All rights reserved."""
import math
from typing import List, Optional

import torch
from torch.optim.lr_scheduler import _LRScheduler

# pyright: reportGeneralTypeIssues=false
# pylint: disable=invalid-name


class CosineAnnealingWarmUpRestarts(_LRScheduler):
    """Scheduler for CosineAnnelingWarmUp with decreasing method implemented"""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        T_0: int,
        T_mult: int = 1,
        eta_max: float = 0.1,
        T_up: int = 0,
        gamma: float = 1.0,
        last_epoch: int = -1,
    ) -> None:
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError(f"Expected positive integer T_0, but got {T_0}")
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError(f"Expected integer T_mult >= 1, but got {T_mult}")
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError(f"Expected positive integer T_up, but got {T_up}")
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Get current lr

        Returns:
            List[float]: LRs
        """
        if self.T_cur == -1:
            return self.base_lrs
        if self.T_cur < self.T_up:
            return [
                (self.eta_max - base_lr) * self.T_cur / self.T_up + base_lr
                for base_lr in self.base_lrs
            ]
        return [
            base_lr
            + (self.eta_max - base_lr)
            * (
                1
                + math.cos(
                    math.pi * (self.T_cur - self.T_up) / (self.T_i - self.T_up)
                )
            )
            / 2
            for base_lr in self.base_lrs
        ]

    def step(self, epoch: Optional[int] = None) -> None:
        """Step scheduler

        Args:
            epoch (Optional[int], optional):
                Current epoch. Defaults to None.
        """
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(
                        math.log(
                            (epoch / self.T_0 * (self.T_mult - 1) + 1),
                            self.T_mult,
                        )
                    )
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult**n - 1) / (
                        self.T_mult - 1
                    )
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch

        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr
