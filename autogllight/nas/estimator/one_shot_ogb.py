import torch
import torch.nn.functional as F
from autogllight.utils.evaluation import Acc

from ..space import BaseSpace
from .base import BaseEstimator


class OneShotOGBEstimator(BaseEstimator):
    """
    One shot estimator on ogb data

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

    def infer(self, model: BaseSpace, dataloader, *args, **kwargs):
        device = next(model.parameters()).device
        y_true = []
        y_pred = []
        for batch in dataloader:
            batch = batch.to(device)
            _, pred, _, _ = model(batch)
            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

        y_true = torch.cat(y_true, dim=0).float()
        y_pred = torch.cat(y_pred, dim=0)

        loss = getattr(F, self.loss_f)(y_pred, y_true).item()

        y_true = y_true.view(-1).numpy()
        y_pred = y_pred.view(-1).numpy()

        metrics = {
            eva.get_eval_name(): eva.evaluate(y_pred, y_true) for eva in self.evaluation
        }
        return metrics, loss
