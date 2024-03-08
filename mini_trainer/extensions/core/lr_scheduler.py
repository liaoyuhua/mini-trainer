from torch.optim import lr_scheduler

from mini_trainer.extensions import Extension
from mini_trainer.extensions.base import Priority

LR_SCHEDULERS = ['OneCycleLR', 'CosineAnnealingLR', 'ReduceLROnPlateau', 'StepLR', 'MultiStepLR',
                 'ExponentialLR', 'CyclicLR', 'CosineAnnealingWarmRestarts']


class LrScheduler(Extension):
    def __init__(self, scheduler: str, config: dict = None):
        super().__init__()
        self.scheduler = scheduler
        self.config = config

        self.set_priority(Priority.HIGHEST)

    def _init_scheduler_obj(self, scheduler: str, optimizer, config: dict):
        if scheduler not in LR_SCHEDULERS:
            raise ValueError(f"Invalid scheduler: {scheduler}. Choose from {LR_SCHEDULERS}")
        self.scheduler = getattr(lr_scheduler, scheduler)(optimizer, **config)

    def on_init_end(self):
        # initialize scheduler object
        self._init_scheduler_obj(self.scheduler, self.trainer.optimizer, self.config)
        # set scheduler to trainer
        self.set_value_to_trainer('scheduler', self.scheduler)

    def on_train_epoch_end(self):
        self.trainer.scheduler.step()
