"""
Base estimator of NAS
"""

from abc import abstractmethod
from ..space import BaseSpace
from typing import Tuple
import torch.nn.functional as F
import torch


class BaseEstimator:
    """
    The estimator of NAS model.

    Parameters
    ----------
    loss_f: callable
        Default loss function for evaluation

    evaluation: list of autogl.module.train.evaluation.Evaluation
        Default evaluation metric
    """

    def __init__(self, loss_f, evaluation):
        """
        Initial estimator

        Parameters
        ----------
        loss_f: None || function(BaseSpace, dataset, mask)->torch.Tensor
            Get the loss function, should be differentiable

        evaluation: None || function(BaseSpace, dataset, mask)->float
            Get an evaluation function

        """
        self.loss_f = loss_f
        self.evaluation = evaluation

    def setLossFunction(self, loss_f: str):
        self.loss_f = loss_f

    def setEvaluation(self, evaluation):
        self.evaluation = evaluation

    @abstractmethod
    def infer(
        self, model: BaseSpace, dataset, mask="train", *args, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the loss and metrics of given model on given dataset using
        specified masks.

        Parameters
        ----------
        model: autogl.module.nas.space.BaseSpace
            The model in space.

        dataset: any
            The dataset to perform infer

        mask: str
            The mask to evalute on dataset

        Return
        ------
        metrics: list of float
            the metrics on given datasets.
        loss: torch.Tensor
            the loss on given datasets. Note that loss should be differentiable.
        """
        raise NotImplementedError()
