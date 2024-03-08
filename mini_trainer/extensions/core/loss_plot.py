from matplotlib import pyplot as plt

from mini_trainer.extensions import Extension


class LossPlot(Extension):
    """
    Plot loss curve. Refresh automatically in the end of each epoch or iter.
    For

    """

    def __init__(self, plot_n_periods: int = 0) -> None:
        """
        Args:
            plot_n_periods: Number of periods (epochs or iters) to plot the loss, 0 to disable.
        """
        super().__init__()
        self.plot_n_periods = plot_n_periods

    @staticmethod
    def _plot_loss(train_loss: list, val_loss: list = None, block=False):
        """
        Plot loss curve. Refresh automatically.
        """
        plt.plot(train_loss, label='train_loss')
        if val_loss or len(val_loss) > 0:
            plt.plot(val_loss, label='val_loss')
        plt.legend()
        plt.show(block=block)
        plt.close()

    def on_train_end(self):
        self._plot_loss(self.trainer.state['epoch_train_loss'][-1], self.trainer.state['epoch_val_loss'][-1],
                        block=True)

    def on_train_epoch_end(self):
        if not self.trainer.state['validation'] and self.trainer.state['by_epoch'] and self.every_n_epochs(
                self.plot_n_periods):
            self._plot_loss(self.trainer.state['epoch_train_loss'][-1])

    def on_validation_end(self):
        if self.trainer.state['validation'] and self.trainer.state['by_epoch'] and self.every_n_epochs(
                self.plot_n_periods):
            self._plot_loss(self.trainer.state['epoch_train_loss'][-1], self.trainer.state['epoch_val_loss'][-1])

    def on_train_batch_end(self):
        if not self.trainer.state['validation'] and not self.trainer.state['by_epoch'] and self.every_n_epochs(
                self.plot_n_periods):
            self._plot_loss(self.trainer.state['epoch_train_loss'][-1])

    def on_validation_batch_end(self):
        if self.trainer.state['validation'] and not self.trainer.state['by_epoch'] and self.every_n_epochs(
                self.plot_n_periods):
            self._plot_loss(self.trainer.state['epoch_train_loss'][-1], self.trainer.state['epoch_val_loss'][-1])
