"""
Base class for algorithm
"""
import torch
from abc import abstractmethod
from autogllight.utils import get_device


class BaseNAS:
    """
    Base NAS algorithm class

    Parameters
    ----------
    device: str or torch.device
        The device of the whole process
    """

    def __init__(self, device="auto") -> None:
        self.device = get_device(device)
        self.selection = None

    def to(self, device):
        """
        Change the device of the whole NAS search process

        Parameters
        ----------
        device: str or torch.device
        """
        self.device = get_device(device)

    @abstractmethod
    def search(self, space, dataset, estimator, return_model=True):
        """
        The search process of NAS.

        Parameters
        ----------
        space : autogllight.nas.space.BaseSpace
            The search space. Constructed following nni.
        dataset : any
            Dataset to perform search on.
        estimator : autoglight.nas.estimator.BaseEstimator
            The estimator to compute loss & metrics.

        Returns
        -------
        model: autogllight.nas.space.BaseSpace || dict
            The searched model (return_model=True) or the selection (return_model=False).
        """
        raise NotImplementedError()

    def get_selection(self):
        return self.selection
