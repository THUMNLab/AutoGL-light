import numpy as np
import typing as _typing

import torch
import pickle

import logging

def get_logger(name):
    """
    Get the logger of given name

    Parameters
    ----------
    name: str
        The name of logger

    Returns
    -------
    logger: Logger
        The logger generated
    """
    return logging.getLogger(name)


LOGGER_ES = get_logger("early-stopping")



class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self,
        patience=7,
        verbose=False,
        delta=0,
        path="checkpoint.pt",
        trace_func=LOGGER_ES.info,
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = 100 if patience is None else patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            if self.verbose is True:
                self.trace_func(
                    f"EarlyStopping counter: {self.counter} out of {self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        self.best_param = pickle.dumps(model.state_dict())
        # torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

    def load_checkpoint(self, model):
        """Load models"""
        if hasattr(self, "best_param"):
            model.load_state_dict(pickle.loads(self.best_param))
        else:
            LOGGER_ES.warn("try to load checkpoint while no checkpoint is saved")

