import torch.nn.functional as F
from ..space import BaseSpace
from .base import BaseEstimator
from autogllight.utils.evaluation import Acc
from autogllight.utils.backend.op import *


class TrainScratchEstimator(BaseEstimator):
    """
    Train scratch estimator.

    Train the model to get estimations.

    Parameters
    ----------
    trainer : str
        The trainer to train the model
    evaluation : list of Evaluation
        The evaluation metrics in module/train/evaluation
    """

    def __init__(self, trainer, evaluation=[Acc()]):
        super().__init__(None, evaluation)
        self.trainer = trainer
        self.evaluation = evaluation        

    def infer(self, model: BaseSpace, dataset, mask="train", *args, **kwargs):
        metrics, loss = self.trainer(model, dataset, mask, self.evaluation, *args, **kwargs)
        return metrics, loss
