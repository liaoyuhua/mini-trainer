# Mini Trainer for PyTorch

This is a mini Trainer for PyTorch ecosystem. Particularly suitable for research and experiments because of the following advantages:

* Fully transparent and retraceable training process
* Low code volume for easy debugging
* Meets the main requirements for model training and evaluating

Main features:

* Pipeline for model training and evaluating
* Checkpoint
* Earlystopping
* Logging based on json file

## Installation
```bash
pip install minitrainer
```

## Quick Start
Below is two examples for starting using mini-trainer. First is classic image classification task and another is house price regression. Both of them are complete deep learning project, and you can learn how the basic usage and main APIs of this project.

[MINST Classification](https://www.runoob.com)

[House Sale Price Prediction](https://www.runoob.com)

## Main Functions and APIs

Initialization: Trainer()

|      API       |         Type          |                             Desc                             |
| :------------: | :-------------------: | :----------------------------------------------------------: |
|     model      |       nn.Module       |                   A model object to train.                   |
|   save_path    |          str          |       Path to save checkpoints/loss plot/log file/etc.       |
|   optimizer    | torch.optim.optimizer |                Optimizer class, default Adam                 |
|       lr       |         float         |               Learning rate, default **1e-3**                |
|      loss      |       callable        |              Loss function, default **L1 loss**              |
|     device     |          str          |   Device type, default **"auto"**. ["auto", "cpu", "cuda"]   |
| early_stopping |         bool          |             Whether early stopping, default True             |
| stop_patience  |          int          |                 Stop patience, default **7**                 |
|   stop_mode    |          str          | Stop mode. For example, if you use MSE to test you model,  this argument should be "min" while this should be "max" for Accuracy. default **"min"** |

Model Training: fit()

|       API        |            Type             |                             Desc                             |
| :--------------: | :-------------------------: | :----------------------------------------------------------: |
| train_dataloader | torch.utils.data.DataLoader |                     Training dataloader.                     |
|  val_dataloader  | torch.utils.data.DataLoader |                    Validation dataloader.                    |
|      epochs      |             int             |               Number of epochs, default **50**               |
|     prog_bar     |            bool             | Whether display progress bar to monitor training process, default  **True** |

Predicting: predict()

|       API       |            Type             |    Desc     |
| :-------------: | :-------------------------: | :---------: |
| test_dataloader | torch.utils.data.DataLoader | Dataloader. |

Result saving: log()

| API  | Type |                             Desc                             |
| :--: | :--: | :----------------------------------------------------------: |
| log  | dict | Anything you want to record to log file, saved as a dictionary. It is very useful for research experiments in which you can record experiment start time, version, key hyperparameters, etc. |

Plot loss curve: plot_loss()

| API  | Type |            Desc            |
| :--: | :--: | :------------------------: |
| save | bool | Whether to save loss plot. |
