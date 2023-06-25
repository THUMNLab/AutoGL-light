# codes in this file are reproduced from https://github.com/microsoft/nni with some changes.
import torch
import torch.nn as nn
import logging

from .base import BaseNAS
from ..space import (
    BaseSpace,
    replace_layer_choice,
    replace_input_choice,
    get_module_order,
    sort_replaced_module,
    PathSamplingInputChoice,
    PathSamplingLayerChoice,
    apply_fixed_architecture,
)
from tqdm import tqdm
from datetime import datetime
import numpy as np
from .rl_utils import ReinforceController, ReinforceField

LOGGER = logging.getLogger(__name__)


class RL(BaseNAS):
    """
    RL in GraphNas.

    Parameters
    ----------
    num_epochs : int
        Number of epochs planned for training.
    device : torch.device
        ``torch.device("cpu")`` or ``torch.device("cuda")``.
    log_frequency : int
        Step count per logging.
    grad_clip : float
        Gradient clipping. Set to 0 to disable. Default: 5.
    entropy_weight : float
        Weight of sample entropy loss.
    skip_weight : float
        Weight of skip penalty loss.
    baseline_decay : float
        Decay factor of baseline. New baseline will be equal to ``baseline_decay * baseline_old + reward * (1 - baseline_decay)``.
    ctrl_lr : float
        Learning rate for RL controller.
    ctrl_steps_aggregate : int
        Number of steps that will be aggregated into one mini-batch for RL controller.
    ctrl_steps : int
        Number of mini-batches for each epoch of RL controller learning.
    ctrl_kwargs : dict
        Optional kwargs that will be passed to :class:`ReinforceController`.
    n_warmup : int
        Number of epochs for training super network.
    model_lr : float
        Learning rate for super network.
    model_wd : float
        Weight decay for super network.
    disable_progress: boolean
        Control whether show the progress bar.
    """

    def __init__(
        self,
        num_epochs=5,
        device="auto",
        log_frequency=None,
        grad_clip=5.0,
        entropy_weight=0.0001,
        skip_weight=0.8,
        baseline_decay=0.999,
        ctrl_lr=0.00035,
        ctrl_steps_aggregate=20,
        ctrl_kwargs=None,
        n_warmup=100,
        model_lr=5e-3,
        model_wd=5e-4,
        disable_progress=False,
        weight_share=True,
    ):
        super().__init__(device)
        self.num_epochs = num_epochs
        self.log_frequency = log_frequency
        self.entropy_weight = entropy_weight
        self.skip_weight = skip_weight
        self.baseline_decay = baseline_decay
        self.baseline = 0.0
        self.ctrl_steps_aggregate = ctrl_steps_aggregate
        self.grad_clip = grad_clip
        self.ctrl_kwargs = ctrl_kwargs
        self.ctrl_lr = ctrl_lr
        self.n_warmup = n_warmup
        self.model_lr = model_lr
        self.model_wd = model_wd
        self.disable_progress = disable_progress
        self.weight_share = weight_share

    def search(self, space: BaseSpace, dset, estimator):
        self.model = space
        self.dataset = dset  # .to(self.device)
        self.estimator = estimator
        # replace choice
        self.nas_modules = []

        k2o = get_module_order(self.model)
        replace_layer_choice(self.model, PathSamplingLayerChoice, self.nas_modules)
        replace_input_choice(self.model, PathSamplingInputChoice, self.nas_modules)
        self.nas_modules = sort_replaced_module(k2o, self.nas_modules)

        # to device
        self.model = self.model.to(self.device)
        # fields
        self.nas_fields = [
            ReinforceField(
                name,
                len(module),
                isinstance(module, PathSamplingLayerChoice) or module.n_chosen == 1,
            )
            for name, module in self.nas_modules
        ]
        self.controller = ReinforceController(
            self.nas_fields, **(self.ctrl_kwargs or {})
        )
        self.ctrl_optim = torch.optim.Adam(
            self.controller.parameters(), lr=self.ctrl_lr
        )
        # train
        with tqdm(range(self.num_epochs), disable=self.disable_progress) as bar:
            for i in bar:
                l2 = self._train_controller(i)
                bar.set_postfix(reward_controller=l2)

        selection = self.export()
        arch = space.parse_model(selection)
        return arch

    def _train_controller(self, epoch):
        self.model.eval()
        self.controller.train()
        self.ctrl_optim.zero_grad()
        rewards = []
        with tqdm(
            range(self.ctrl_steps_aggregate), disable=self.disable_progress
        ) as bar:
            for ctrl_step in bar:
                self._resample()
                metric, loss = self._infer(mask="val")
                reward = metric["acc"]
                bar.set_postfix(loss=loss.item(), **metric)
                LOGGER.debug(f"{self.selection}\n{metric},{loss}")
                rewards.append(reward)
                if self.entropy_weight:
                    reward += (
                        self.entropy_weight * self.controller.sample_entropy.item()
                    )
                self.baseline = self.baseline * self.baseline_decay + reward * (
                    1 - self.baseline_decay
                )
                loss = self.controller.sample_log_prob * (reward - self.baseline)
                if self.skip_weight:
                    loss += self.skip_weight * self.controller.sample_skip_penalty
                loss /= self.ctrl_steps_aggregate
                loss.backward()

                if (ctrl_step + 1) % self.ctrl_steps_aggregate == 0:
                    if self.grad_clip > 0:
                        nn.utils.clip_grad_norm_(
                            self.controller.parameters(), self.grad_clip
                        )
                    self.ctrl_optim.step()
                    self.ctrl_optim.zero_grad()

                if (
                    self.log_frequency is not None
                    and ctrl_step % self.log_frequency == 0
                ):
                    LOGGER.debug(
                        "RL Epoch [%d/%d] Step [%d/%d]  %s",
                        epoch + 1,
                        self.num_epochs,
                        ctrl_step + 1,
                        self.ctrl_steps_aggregate,
                    )
        return sum(rewards) / len(rewards)

    def _resample(self):
        result = self.controller.resample()
        if self.weight_share:
            for name, module in self.nas_modules:
                module.sampled = result[name]
        else:
            self.arch = self.model.parse_model(result)
        self.selection = result

    def export(self):
        self.controller.eval()
        with torch.no_grad():
            return self.controller.resample()

    def _infer(self, mask="train"):
        if self.weight_share:
            metric, loss = self.estimator.infer(self.model, self.dataset, mask=mask)
        else:
            metric, loss = self.estimator.infer(
                self.arch._model, self.dataset, mask=mask
            )
        return metric, loss
