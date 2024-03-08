from mini_trainer.trainer import Trainer
from mini_trainer.templates import l1_extensions, l2_extensions, l3_extensions
from mini_trainer.extensions import Checkpoint, EarlyStopping, LrScheduler, LossPlot

__all__ = [
    "Trainer",
    "l1_extensions",
    "l2_extensions",
    "l3_extensions",
    "Checkpoint",
    "EarlyStopping",
    "LrScheduler",
    "LossPlot",
]

__version__ = "0.2.3"


def get_env_info():
    """Get basic environment information."""
    import torch

    msg = r"""
.___.
  |  ._. _.*._ *._  _
  |  [  (_]|[ )|[ )(_]
                   ._|
    """
    msg += (
        "\nVersion Information: "
        f"\n\tmini_trainer: {__version__}"
        f"\n\tPyTorch: {torch.__version__}"
    )
    return msg
