from typing import Tuple, Optional, List
from warnings import warn
import torch
import numpy as np

from mini_trainer.extensions import Extension


class EarlyStopping(Extension):
    stop_modes = ("min", "max")

    def __init__(
            self,
            min_delta: float = 0.0,
            stopping_threshold: float = None,
            divergence_threshold: float = None,
            patience: int = 3,
            mode: str = "min",
            metrics_weights: Optional[List[float]] = None,
            verbose: bool = True
    ) -> None:
        """
        Args:
            min_delta: Minimum change in the monitored metric to qualify as an improvement.
            stopping_threshold: Stop mini_trainer when the monitored metric reaches this threshold.
            divergence_threshold: Stop mini_trainer when the monitored metric reaches this threshold.
            patience: Number of records to wait for the monitored metric to improve.
            mode: One of {"min", "max"}. In "min" mode, mini_trainer will stop when the monitored metric stops decreasing.
                In "max" mode, mini_trainer will stop when the monitored metric stops increasing.
            metrics_weights: Weights for each metric in the monitored metrics. If None, all metrics are weighted equally.
            verbose: Whether to print the early stopping messages.
        """
        super().__init__()
        if min_delta < 0:
            warn(
                f"EarlyStopping min_delta {min_delta} is negative, convert to absolute value."
            )

        self.min_delta = abs(min_delta)
        self.stopping_threshold = stopping_threshold
        self.divergence_threshold = divergence_threshold
        self.patience = patience
        self.metrics_weights = [1.0] if metrics_weights is None else metrics_weights

        assert mode in self.stop_modes, f"EarlyStopping mode {mode} is unknown."

        if mode == "min":
            self.monitor_op = torch.lt
        elif mode == "max":
            self.monitor_op = torch.gt

        self.mode = mode
        self.mode_order = {"min": "<", "max": ">"}

        self.min_delta *= 1 if self.monitor_op == torch.gt else -1
        torch_inf = torch.tensor(np.Inf)
        self.best_score = torch_inf if self.monitor_op == torch.lt else -torch_inf

        self.verbose = verbose

    def _run_early_stopping_check(self):
        """
        Checks whether the early stopping condition is met and if so tells the trainer to stop the mini_trainer.
        """
        cur_val_metrics = torch.tensor(self.trainer.state['epoch_val_metrics'][-1], dtype=torch.float32)
        if len(cur_val_metrics) != len(self.metrics_weights):
            raise ValueError(
                f"Number of monitored metrics {len(cur_val_metrics)} does not match the number of weights"
                f" {len(self.metrics_weights)}."
            )
        cur_val_metrics = torch.dot(cur_val_metrics, torch.tensor(self.metrics_weights, dtype=cur_val_metrics[0].dtype))

        should_stop, reason = self._evaluate_stopping_criteria(cur_val_metrics)

        # stop every ddp process if any world process decides to stop
        self.trainer.state['stop_mini_trainer'] = should_stop
        if should_stop:
            # larger of self.trainer.state['cur_epoch_num'] and self.trainer.state['cur_iter_num']
            self.stopped_period = max(self.trainer.state['cur_epoch_num'], self.trainer.state['cur_iter_num'])
        if reason and self.verbose and hasattr(self.trainer, "logger"):
            self.trainer.logger.info(reason)

    def _evaluate_stopping_criteria(self, current: torch.Tensor):
        should_stop = False
        reason = None
        if not torch.isfinite(current):
            should_stop = True
            reason = (
                f"Monitored metric = {current} is not finite."
                f" Previous best value was {self.best_score:.3f}. Signaling Trainer to stop."
            )
        elif self.stopping_threshold is not None and self.monitor_op(current, self.stopping_threshold):
            should_stop = True
            reason = (
                "Stopping threshold reached:"
                f" {current} {self.mode_order[self.mode]} {self.stopping_threshold}."
                " Signaling Trainer to stop."
            )
        elif self.divergence_threshold is not None and self.monitor_op(-current, -self.divergence_threshold):
            should_stop = True
            reason = (
                "Divergence threshold reached:"
                f" {current} {self.mode_order[self.mode]} {self.divergence_threshold}."
                " Signaling Trainer to stop."
            )
        elif self.monitor_op(current - self.min_delta, self.best_score.to(current.device)):
            should_stop = False
            reason = self._improvement_message(current)
            self.best_score = current
            self.wait_count = 0
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                should_stop = True
                reason = (
                    f"Monitored metric did not improve in the last {self.wait_count} records."
                    f" Best score: {self.best_score:.3f}. Signaling Trainer to stop."
                )

        return should_stop, reason

    def _improvement_message(self, current: torch.Tensor):
        """
        Formats a log message that informs the user about an improvement in the monitored score.
        """
        if torch.isfinite(self.best_score):
            msg = (
                f"Metric improved by {abs(self.best_score - current):.3f} >="
                f" min_delta = {abs(self.min_delta)}. New best score: {current:.3f}"
            )
        else:
            msg = f"Metric improved. New best score: {current:.3f}"
        return msg

    def on_validation_end(self):
        """
        Called by `on_validation_end`.
        """
        self._run_early_stopping_check()
