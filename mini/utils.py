import torch
import numpy as np


class EarlyStopping:
    stop_ops = {
        "min": np.greater,
        "max": np.less
    }

    def __init__(
        self,
        patience=7,
        verbose=False,
        delta=0,
        mode="min",
        stop=True
    ):
        assert mode in ("min", "max"), f"only min/max is supported, mode {mode} is invalid"
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = np.Inf if mode=="min" else -np.Inf
        self.op = self.stop_ops[mode]
        self.early_stop = False
        self.best_model = None
        self.delta = delta
        self.mode = mode
        self.stop = stop

    def __call__(self, val_loss, model, path):
        if self.op(val_loss, self.best_score + self.delta):
            if self.stop:
                self.counter += 1
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
        else:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss ({self.best_score:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path+'/checkpoint.pth')
        self.best_score = val_loss
        self.best_model = model


def move_to_device(x, device):
    if isinstance(device, str):
        device = torch.device(device)
    if isinstance(x, dict):
        for name in x.keys():
            x[name] = move_to_device(x[name], device=device)
    elif isinstance(x, torch.Tensor) and x.device != device:
        x = x.to(device)
    elif isinstance(x, (list, tuple)) and x[0].device != device:
        x = [move_to_device(xi, device=device) for xi in x]
    return x