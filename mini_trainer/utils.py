from typing import List, Tuple, Dict, Union
from torch import Tensor


def move_to_device(batch: Union[Tensor, List[Tensor], Tuple[Tensor], Dict[str, Tensor]], device: str):
    """
    Move the input to the specified device.
    """
    if isinstance(batch, Tensor):
        return batch.to(device)
    elif isinstance(batch, tuple):
        return tuple(t.to(device) for t in batch)
    elif isinstance(batch, dict):
        return {k: v.to(device) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [t.to(device) for t in batch]
    else:
        raise ValueError("batch must be one of Tensor, Tuple, Dict")
