import torch
from torch.nn import functional as F
from functools import wraps
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from mini.utils import EarlyStopping, move_to_device
import matplotlib.pyplot as plt


def timer(keyword):
    def timeit(func):
        @wraps(func)
        def timeit_wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            total_time = end_time - start_time
            print(f'{keyword} took {total_time/60:.2f} minutes')
            return result
        return timeit_wrapper
    return timeit


class Trainer:
    devices = ("cuda", "cpu")

    def __init__(
        self,
        model,
        save_path,
        optimizer=torch.optim.Adam,
        lr=1e-3,
        loss=F.l1_loss,
        device="auto",
        early_stopping=True,
        stop_patience=7,
        stop_mode="min",
    ) -> None:
        """
        Args:
            model: model object

        """
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device in self.devices:
            self.device = device
        else:
            raise ValueError(f"Invaid device type: {device}")

        self.model = model.to(self.device)
        self.train_loss = []
        self.val_loss = []
        self.test_mae = None
        self.test_mse = None
        self.best_val_loss = None
        self.optimizer = optimizer(self.model.parameters(), lr=lr)
        self.loss = loss

        self.early_stopping = early_stopping
        self.__init_early_stopping(early_stopping, stop_patience, stop_mode)
        
        self.save_path = Path(save_path)

        self.ckp_path = self.save_path/"checkpoints"
        self.ckp_path.mkdir(parents=True, exist_ok=True)

    def __init_early_stopping(self, if_stop, stop_patience, stop_mode):
        if if_stop:
            self.es = EarlyStopping(patience=stop_patience, mode=stop_mode)
        else:
            self.es = EarlyStopping(mode=stop_mode, stop=False)

    @timer("training")
    def fit(self, train_dataloader, val_dataloader, epochs=50, prog_bar=True):
        for i in range(epochs):
            epoch_train_loss = []
            self.model.train()
            with tqdm(train_dataloader, unit="batch", disable=(not prog_bar)) as tepoch:
                for x, y in tepoch:
                    tepoch.set_description(f"Epoch {i}")
                    x = move_to_device(x, self.device)
                    y = move_to_device(y, self.device)
                    out = self.model(x)

                    loss = self.loss(out, y)
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    epoch_train_loss.append(loss.item())
                    tepoch.set_postfix(train_loss=np.round(np.mean(epoch_train_loss), 4))

            val_loss = self.val(val_dataloader)
            tepoch.set_postfix(train_loss=np.round(np.mean(epoch_train_loss), 3), 
                            valid_loss=np.round(val_loss, 3))
            self.train_loss.append(np.mean(epoch_train_loss))
            self.val_loss.append(val_loss)
            
            self.es(val_loss, self.model, str(self.ckp_path))
            self.best_val_loss = self.es.best_score
            if self.es.early_stop:
                break
    
    @torch.no_grad()
    def val(self, val_dataloader):
        losses = []
        self.model.eval()
        for x, y in val_dataloader:
            x = move_to_device(x, self.device)
            y = move_to_device(y, self.device)
            out = self.model(x)
            loss = self.loss(out, y)
            losses.append(loss.item())
        loss_value = np.mean(losses)
        return loss_value 
    
    @timer("predicting")
    def predict(self, test_dataloader):
        model = self.es.best_model
        model.eval()
        pred = []
        with torch.no_grad():
            for x, y in test_dataloader:
                x = move_to_device(x, self.device)
                y = move_to_device(y, self.device)
                out = model(x)
                pred.append(out)

        return torch.cat(pred, dim=0)
            
    def log(self, log={}):
        trainer_log = {
            "save_path": str(self.save_path),
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "best_val_loss": self.best_val_loss,
            "ckp_path": str(self.ckp_path),
            "device": self.device
        }
        
        log.update(trainer_log)
        
        log_json = json.dumps(log, indent=4,separators=(',',':'))
        f=open(self.save_path/"log.json","w")
        f.write(log_json)
        f.close()
    
    def plot_loss(self, save=False):
        assert len(self.train_loss) == len(self.val_loss), "length of train_loss is not equal to that of val_loss"

        plt.plot(list(range(1, len(self.train_loss)+1)), self.train_loss, color='green', label="train_loss")
        plt.plot(list(range(1, len(self.val_loss)+1)), self.val_loss, color='red', label="valid_loss")
        plt.legend()
        plt.show()

        if save:
            plt.savefig(self.save_path/"loss.png")