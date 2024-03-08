# import pytest
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from mini_trainer import *
from tests.mnist import SimpleCNN, train_loader, test_loader


def test_trainer():
    model = SimpleCNN()

    checkpoint = Checkpoint(save_n_periods=1, max_checkpoints=1)
    loss_plot = LossPlot(1)
    lr_scheduler = LrScheduler('OneCycleLR', {'max_lr': 0.1, 'epochs': 1, 'steps_per_epoch': 10})
    early_stopping = EarlyStopping(1)

    optimizer = Adam(model.parameters(), lr=0.001)

    def loss(output, batch):
        return CrossEntropyLoss()(output, batch[1])

    def accuracy_fn(output, batch):
        return (output.argmax(1) == batch[1]).float().mean()

    trainer = Trainer(
        model,
        optimizer,
        loss,
        accuracy_fn,
        extensions=[checkpoint, loss_plot, lr_scheduler, early_stopping]
    )

    trainer.fit(train_loader, test_loader, 3)


