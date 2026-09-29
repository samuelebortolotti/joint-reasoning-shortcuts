# config module

import random
import torch
import numpy as np
import os

# Global variable to store the preferred device
PREFERRED_DEVICE = None

def set_preferred_device(args):
    """Sets the preferred device based on command-line arguments."""
    global PREFERRED_DEVICE
    if args.device:
        PREFERRED_DEVICE = args.device

def get_device() -> torch.device:
    """Returns the GPU device if available else CPU.

    Returns:
        device: device
    """
    global PREFERRED_DEVICE
    if PREFERRED_DEVICE:
        if PREFERRED_DEVICE.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError(f"CUDA is not available, but '{PREFERRED_DEVICE}' was specified.")
        return torch.device(PREFERRED_DEVICE)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def base_path() -> str:
    """Returns the base bath where to log accuracies and tensorboard data.

    Returns:
        base_path (str): base path
    """
    return "./data/"


def set_random_seed(seed: int) -> None:
    """Sets the seeds at a certain value.

    Args:
        param seed: the value to be set

    Returns:
        None: This function does not return a value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_path(path) -> None:
    """Create path function, create folder if it does not exists

    Args:
        path (str): path value

    Returns:
        None: This function does not return a value.
    """
    if not os.path.exists(path):
        os.makedirs(path)
