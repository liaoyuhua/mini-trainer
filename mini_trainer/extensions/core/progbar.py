from mini_trainer.extensions import Extension

from tqdm import tqdm


class Progbar(Extension):
    def __init__(self) -> None:
        super().__init__()

    def on_train_batch_end(self) -> None:
        pass

    def on_validation_end(self) -> None:
        pass
