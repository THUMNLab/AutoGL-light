import numpy as np
import random
import os
import torch


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False


import torch
from typing import Union


def get_device(device: Union[str, torch.device]):
    """
    Get device of passed argument. Will return a torch.device based on passed arguments.
    Can parse auto, cpu, gpu, cpu:x, gpu:x, etc. If auto is given, will automatically find
    available devices.


    Parameters
    ----------
    device: ``str`` or ``torch.device``
        The device to parse. If ``auto`` if given, will determine automatically.

    Returns
    -------
    device: ``torch.device``
        The parsed device.
    """
    assert isinstance(
        device, (str, torch.device)
    ), "Only support device of str or torch.device, get {} instead".format(device)
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
