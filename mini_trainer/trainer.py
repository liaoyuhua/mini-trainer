from typing import List, Union, Callable
from warnings import warn
import numpy as np
import torch
from torch.optim import Optimizer

from mini_trainer.extensions.base import Extension
from mini_trainer.utils import move_to_device


class Trainer:
    """
    Single GPU mini_trainer.
    """

    def __init__(
            self,
            model: torch.nn.Module,  # in: Tensor, out: Tensor
            optimizer: Optimizer,
            loss_fn: Callable,  # in: Tensor, Tensor, out: torch.Tensor
            metric_fn: Union[Callable, List[Callable]],  # in: Tensor, Tensor, out: float
            device: str = 'auto',
            extensions: Union[Extension, List[Extension], None] = None,
    ):
        self._extensions = []
        self._register_extensions(extensions)
        self._run_extension('on_init_start')
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self._init_device(device)

        # trainer state: information and objects that can be accessed by extensions
        # don't change these attributes directly to avoid unexpected behaviors
        # if you want to develop a new extension, you can access these attributes confidently
        self.state = {
            'validation': False,  # whether validation is performed
            'by_epoch': True,  # by_epoch or by_iter
            'cur_iter_num': 0,  # iter starts from 1,
            'cur_epoch_num': 0,  # epoch starts from 1
            'cur_train_batch_idx': 0,  # batch_idx starts from 0
            'stop_mini_trainer': False,  # max_epochs or max_iters or early stopping
            'batch_train_loss': [],  # reset every epoch (by_epoch) or every eval_n_periods (by_iter)
            'batch_val_loss': [],  # reset every epoch (by_epoch) or every eval_n_periods (by_iter)
            'batch_val_metrics': [],  # reset every epoch (by_epoch) or every eval_n_periods (by_iter)
            'epoch_train_loss': [],  # record each epoch (by epoch) or every eval_n_periods (by_iter)
            'epoch_val_loss': [],  # record each epoch (by epoch) or every eval_n_periods (by_iter)
            'epoch_val_metrics': [],  # record each epoch (by epoch) or every eval_n_periods (by_iter)
            'device': self.device,
            'model': self.model,
            'optimizer': self.optimizer,
            'loss_fn': self.loss_fn,
            'metric_fn': self.metric_fn,
        }

        self._run_extension('on_init_end')

    def _init_device(self, device: str):
        assert device in ['auto', 'cpu', 'cuda'], "device must be one of 'auto', 'cpu', 'cuda'"
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

    def _calc_metrics(self, output: torch.Tensor, batch: torch.Tensor):
        if isinstance(self.metric_fn, list):
            metrics = [metric_fn(output, batch) for metric_fn in self.metric_fn]
        else:
            metrics = [self.metric_fn(output, batch)]
        return metrics

    def _reset_batch_loss_metrics(self):
        self.state['batch_train_loss'] = []
        self.state['batch_val_loss'] = []
        self.state['batch_val_metrics'] = []

    def _refresh_train_epoch_loss_metrics(self):
        epoch_loss = sum(self.state['batch_train_loss'])
        self.state['epoch_train_loss'].append(np.mean(epoch_loss))  # average loss per batch

    def fit(self, train_loader, val_loader=None, max_epochs=0, max_iters=0, eval_n_periods=1):
        assert (max_epochs > 0) ^ (max_iters > 0), "Please provide either max_epochs or max_iters, but not both."
        self.state['by_epoch'] = by_epoch = max_epochs > 0
        self.state['validation'] = val_loader is not None
        self.model.to(self.device)
        self.model.train()

        self._run_extension('on_train_start')
        if by_epoch:
            self._train_by_epoch(train_loader, val_loader, max_epochs, eval_n_periods)
        else:
            self._train_by_iter(train_loader, val_loader, max_iters, eval_n_periods)
        self._run_extension('on_train_end')

    def _train_by_epoch(self, train_loader, val_loader, max_epochs, eval_n_periods):
        for epoch in range(max_epochs):
            self.state['cur_epoch_num'] = epoch + 1
            self._run_extension('on_train_epoch_start')
            self._train_epoch(train_loader, val_loader)
            self._run_extension('on_train_epoch_end')
            if self.state['validation'] and (self.state['cur_epoch_num'] % eval_n_periods == 0):
                self.validate(val_loader)
            if self.state['stop_mini_trainer']:
                break
            self._reset_batch_loss_metrics()

    def _train_by_iter(self, train_loader, val_loader, max_iters, eval_n_periods):
        while not self.state['stop_mini_trainer']:  # 2 conditions: max_iters and early stopping
            self._train_epoch(train_loader, val_loader, eval_n_periods, max_iters)

    def _train_epoch(self, train_loader, val_loader, eval_n_periods=1, max_iters=0):
        for idx, batch in enumerate(train_loader):
            self.state['cur_iter_num'] += 1
            self.state['cur_train_batch_idx'] = idx
            batch = move_to_device(batch, self.device)
            self._train_batch(batch, val_loader, eval_n_periods)
            if (self.state['cur_iter_num'] >= max_iters) and not self.state['by_epoch']:
                self.state['stop_mini_trainer'] = True
                break
        if self.state['by_epoch']:
            self._refresh_train_epoch_loss_metrics()

    def _train_batch(self, train_batch, val_loader, eval_n_periods=1):
        self._run_extension('on_train_batch_start')
        self.optimizer.zero_grad()
        output = self.model(train_batch)
        loss = self.loss_fn(output, train_batch)
        self.state['batch_train_loss'].append(loss.item())
        self._run_extension('on_before_backward')
        loss.backward()
        self._run_extension('on_before_optimizer_step')
        self.optimizer.step()

        if self.state['validation'] and (not self.state['by_epoch']) and (
                self.state['cur_iter_num'] % eval_n_periods == 0):
            self.validate(val_loader)
            self._refresh_train_epoch_loss_metrics()  # refresh train loss every eval_n_periods
            self._reset_batch_loss_metrics()  # reset batch loss and metrics every eval_n_periods

        self._run_extension('on_train_batch_end')

    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()
        self._run_extension('on_validation_start')
        for idx, batch in enumerate(val_loader):
            self._validate_batch(batch)
        epoch_loss = sum(self.state['batch_val_loss'])
        epoch_metrics = [sum(m) for m in zip(*self.state['batch_val_metrics'])]
        self.state['epoch_val_loss'].append(epoch_loss / len(val_loader))
        self.state['epoch_val_metrics'].append([m / len(val_loader) for m in epoch_metrics])
        self._run_extension('on_validation_end')

    def _validate_batch(self, batch):
        self._run_extension('on_validation_batch_start')
        output = self.model(batch)
        loss = self.loss_fn(output, batch)
        self.state['batch_val_loss'].append(loss.item())
        self.state['batch_val_metrics'].append(self._calc_metrics(output, batch))
        self._run_extension('on_validation_batch_end')

    def _register_extensions(self, extensions: Union[Extension, List[Extension], None] = None):
        # check if extensions type is correct
        if extensions is None:
            warn("No extensions provided. mini_trainer will run without any extensions.")
        else:
            assert all(
                isinstance(ext, Extension) for ext in list(extensions)), "All extensions must be of type Extension"
            for ext in list(extensions):
                self.register_extension(ext)

    def register_extension(self, extension: Extension):
        """
        Register an extension to the trainer. The extension will be sorted by priority in descending order. If two
        extensions have the same priority, the one registered first will be executed first.
        """
        assert isinstance(extension, Extension)
        extension.post_init(self)

        inserted = False
        for i in range(len(self._extensions) - 1, -1, -1):
            if extension.priority >= self._extensions[i].priority:
                self._extensions.insert(i + 1, extension)
                inserted = True
                break
        if not inserted:
            self._extensions.insert(0, extension)

    def _run_extension(self, event: str):
        for ext in self._extensions:
            getattr(ext, event)()

    @property
    def registered_extensions(self):
        return [ext.name for ext in self._extensions]
