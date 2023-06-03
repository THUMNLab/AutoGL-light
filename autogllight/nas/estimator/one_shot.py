import torch.nn.functional as F
from ..space import BaseSpace
from .base import BaseEstimator
from autogllight.utils.evaluation import Acc
from autogllight.utils.backend.op import *


class OneShotEstimator(BaseEstimator):
    """
    One shot estimator.

    Use model directly to get estimations.

    Parameters
    ----------
    loss_f : str
        The name of loss funciton in PyTorch
    evaluation : list of Evaluation
        The evaluation metrics in module/train/evaluation
    """

    def __init__(self, loss_f="nll_loss", evaluation=[Acc()]):
        super().__init__(loss_f, evaluation)
        self.evaluation = evaluation

    def infer(self, model: BaseSpace, dataset, mask="train", *args, **kwargs):
        device = next(model.parameters()).device
        dset = dataset[0].to(device)
        mask = bk_mask(dset, mask)

        pred = model(dset, *args, **kwargs)[mask]
        label = bk_label(dset)
        y = label[mask]

        loss = getattr(F, self.loss_f)(pred, y)
        probs = F.softmax(pred, dim=1).detach().cpu().numpy()

        y = y.cpu()
        metrics = {
            eva.get_eval_name(): eva.evaluate(probs, y) for eva in self.evaluation
        }
        return metrics, loss
