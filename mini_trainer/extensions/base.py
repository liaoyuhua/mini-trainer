from enum import Enum, unique


@unique
class Priority(Enum):
    HIGHEST = 0
    VERY_HIGH = 10
    HIGH = 30
    ABOVE_NORMAL = 40
    NORMAL = 50
    BELOW_NORMAL = 60
    LOW = 70
    VERY_LOW = 90
    LOWEST = 100


class Extension:
    priority = Priority.NORMAL.value
    trainer = None

    def post_init(self, trainer):
        """
        Called after the extension is initialized. This is the place to set something with the trainer.
        """
        # set the trainer
        self.trainer = trainer

        # check if the extension has valid priority
        if not isinstance(self.priority, int):
            raise ValueError(f"Invalid priority: {self.priority}. Priority must be an integer.")

    def on_init_start(self):
        """
        Called when trainer initialization begins.
        """

    def on_init_end(self):
        """
        Called when trainer initialization ends.
        """

    def on_train_start(self):
        """
        Called when the train begins.
        """

    def on_train_end(self):
        """
        Called when the train ends.
        """

    def on_train_epoch_start(self):
        """
        Called when the train epoch begins.
        """

    def on_train_epoch_end(self):
        """
        Called when the train epoch ends.
        """

    def on_train_batch_start(self):
        """
        Called when the train batch begins.
        """

    def on_train_batch_end(self):
        """
        Called when the train batch ends.
        """

    def on_validation_start(self):
        """
        Called when the validation loop begins.
        """

    def on_validation_end(self):
        """
        Called when the validation loop ends.
        """

    def on_validation_batch_start(self):
        """
        Called when the validation batch begins.
        """

    def on_validation_batch_end(self):
        """
        Called when the validation batch ends.
        """

    def on_before_backward(self):
        """
        Called before ``loss.backward()``.
        """

    def on_before_optimizer_step(self):
        """
        Called before ``optimizer.step()``.
        """

    def set_value_to_trainer(self, name: str, value: any):
        """
        Called to set the value to the trainer. If the trainer already has the attribute, raise an AttributeError.
        This can protect the trainer from being overwritten by the same attribute from different extensions.
        """
        if not hasattr(self.trainer, name):
            setattr(self.trainer, name, value)
        else:
            raise AttributeError(f"Trainer already has attribute {self.name}")

    def set_priority(self, priority: Priority):
        self.priority = priority.value

    def every_n_epochs(self, n):
        """
        If trainer is mini_trainer by epoch, return True if the current epoch number is divisible by n.
        If trainer is mini_trainer by iter, return False (cur_epoch_num is always 0).
        """
        return (self.trainer.state['cur_epoch_num']) % n == 0 if n > 0 else False

    def every_n_iters(self, trainer, n):
        """
        If trainer is mini_trainer by iter, return True if the current iter number is divisible by n.
        If trainer is mini_trainer by epoch, return False (cur_iter_num is always 0).
        """
        return (self.trainer.state['cur_iter_num']) % n == 0 if n > 0 else False

    @property
    def name(self):
        return self.__class__.__name__
