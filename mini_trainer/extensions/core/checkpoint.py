from typing import Optional, List
import pathlib
import torch

from mini_trainer.extensions import Extension


class Checkpoint(Extension):
    """
    Save checkpoint by epoch or iteration.
    If validation exists, best model on validation dataset will be saved. Otherwise
    , last `max_checkpoints` checkpoints will be saved. The checkpoint path looks like
    `work_dir/checkpoints/epoch_{epoch}_val_metric

    Information saved in checkpoint as dict:

    - period: epoch/iteration
    - state_dict: model_state_dict
    - optimizer: optimizer_state_dict
    - train_loss: mini_trainer period loss
    - val_loss: validation period loss (if validation exists)
    - val_metric: validation period metric (if validation exists)
    """

    def __init__(
            self,
            resume_from_checkpoint: Optional[str] = None,
            checkpoint_path: Optional[str] = None,
            save_n_periods: int = 0,
            save_optimizer: bool = True,
            max_checkpoints: int = 0,
            metrics_weights: Optional[List[float]] = None,
            verbose: bool = True
    ) -> None:
        super().__init__()
        self.resume_from_checkpoint = resume_from_checkpoint
        self.checkpoint_path = checkpoint_path
        self.save_n_periods = save_n_periods
        self.save_optimizer = save_optimizer
        self.max_checkpoints = max_checkpoints
        self.verbose = verbose
        self.metrics_weights = metrics_weights
        self._isvalid_checkpoint_path()

    def _isvalid_checkpoint_path(self):
        """
        Check if checkpoint path is valid.
        """
        if self.checkpoint_path is None:
            self.checkpoint_path = 'checkpoints'
        p = pathlib.Path(self.checkpoint_path)
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _load_checkpoint(ckp_path: str):
        """
        Load checkpoint. Return model and optimizer.
        """
        ckp = torch.load(ckp_path)
        model = ckp['state_dict']
        optimizer = ckp['optimizer']
        return model, optimizer

    def _cal_epoch_val_metric(self, epoch_val_metrics: List[float]):
        """
        Calculate weighted sum of validation metrics.
        """
        return sum([m * w for m, w in zip(epoch_val_metrics, self.metrics_weights)])

    def _save_checkpoint(self):
        """
        Save checkpoint.
        """
        period = self.trainer.state['cur_epoch_num'] if self.trainer.state['by_epoch'] else self.trainer.state[
            'cur_iter_num']
        period_train_loss = self.trainer.state['epoch_train_loss'][period - 1] if self.trainer.state['by_epoch'] else \
            self.trainer.state['iter_train_loss'][period - 1]
        period_val_loss = self.trainer.state['epoch_val_loss'][period - 1] if self.trainer.state['validation'] else None
        period_val_metric = self._cal_epoch_val_metric(
            self.trainer.state['epoch_val_metrics'][period - 1]) if self.trainer.state['validation'] else None

        ckp = {
            'period': period,
            'state_dict': self.trainer.model.state_dict(),
            'optimizer': self.trainer.optimizer.state_dict() if self.save_optimizer else None,
            'train_loss': period_train_loss,
            'val_loss': period_val_loss,
            'val_metric': period_val_metric
        }
        if self.trainer.state['by_epoch'] and self.trainer.state['validation']:
            ckp_path = f"{self.checkpoint_path}/epoch_{ckp['period']}_val_metric_{period_val_metric: .4f}.pth"
        elif self.trainer.state['by_epoch'] and not self.trainer.state['validation']:
            ckp_path = f"{self.checkpoint_path}/epoch_{ckp['period']}.pth"
        elif not self.trainer.state['by_epoch'] and self.trainer.state['validation']:
            ckp_path = f"{self.checkpoint_path}/iter_{ckp['period']}_val_metric_{period_val_metric: .4f}.pth"
        else:
            ckp_path = f"{self.checkpoint_path}/iter_{ckp['period']}.pth"

        if self.max_checkpoints > 0 and self.trainer.state['validation']:
            # if max_checkpoints is set, check if the number of checkpoints exceeds max_checkpoints and whether the
            # current checkpoint is better than the worst checkpoint
            ckp_files = sorted(pathlib.Path(self.checkpoint_path).glob('*.pth'))
            if len(ckp_files) >= self.max_checkpoints:
                ckp_files.sort(key=lambda x: float(x.stem.split('_')[-1]))
                worst_ckp = ckp_files[0]
                worst_metric = float(worst_ckp.stem.split('_')[-1])
                if period_val_metric > worst_metric:
                    worst_ckp.unlink()
                else:
                    return

        torch.save(ckp, ckp_path)

        if self.verbose:
            print(f"Checkpoint saved: {ckp_path}")

    def on_init_start(self):
        """
        Resume checkpoint to continue mini_trainer.
        """
        if self.resume_from_checkpoint:
            model, optimizer = self._load_checkpoint(self.resume_from_checkpoint)
            self.trainer.model.load_state_dict(model)
            if optimizer:  # if optimizer is saved, load it
                self.trainer.optimizer.load_state_dict(optimizer)

    def on_init_end(self):
        n_metrics = len(self.trainer.metrics) if isinstance(self.trainer.metric_fn, list) else 1
        self.metrics_weights = [1.0] * n_metrics if self.metrics_weights is None else self.metrics_weights

    def on_train_epoch_end(self):
        if not self.trainer.state['validation'] and self.trainer.state['by_epoch'] and self.every_n_epochs(
                self.save_n_periods):
            self._save_checkpoint()

    def on_train_batch_end(self):
        if not self.trainer.state['validation'] and not self.trainer.state['by_epoch'] and self.every_n_iters(
                self.save_n_periods):
            self._save_checkpoint()

    def on_validation_end(self):
        if self.trainer.state['validation'] and self.every_n_epochs(
                self.save_n_periods):
            self._save_checkpoint()
