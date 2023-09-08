# "Large-Scale Graph Neural Architecture Search" ICML 22'

import random

import torch
import torch.nn.functional as F
from tqdm import trange

from ..estimator.base import BaseEstimator
from ..space import BaseSpace
from .base import BaseNAS


class Gauss(BaseNAS):
    """
    GAUSS trainer.

    Parameters
    ----------
    num_epochs : int
        Number of epochs planned for training.
    device : str or torch.device
        The device of the whole process
    """

    def __init__(
        self,
        num_epochs=100,
        device="auto",
        disable_progress=False,
        args=None,
    ):
        super().__init__(device=device)
        self.device = device
        self.num_epochs = num_epochs
        self.disable_progress = disable_progress
        self.args = args

    def prepare(self, data):
        self.data = data
        # fix random seed of train/val/test split
        random.seed(2022)
        masks = list(range(data.num_nodes))
        random.shuffle(masks)
        fold = int(data.num_nodes * 0.1)
        train_idx = masks[:fold * 6]
        val_idx = masks[fold * 6: fold * 8]
        test_idx = masks[fold * 8:]
        split_idx = {
            'train': torch.tensor(train_idx).long(),
            'valid': torch.tensor(val_idx).long(),
            'test': torch.tensor(test_idx).long()
        }

        for key in split_idx: split_idx[key] = split_idx[key].to(self.device)

        self.train_idx = split_idx['train'].to(self.device)
        self.valid_idx = split_idx['valid'].to(self.device)
        self.test_idx = split_idx['test'].to(self.device)

    def train_graph(
        self,
        optimizer,
        epoch
    ):
        self.space.train()

        optimizer.zero_grad()
        archs = self.space.sampler.samples(self.args.repeat)

        if self.args.use_curriculum:
            judgement = None
            best_acc = 0
            if epoch < self.args.warm_up:
                # min_ratio = args.min_ratio[0]
                min_ratio = epoch / self.args.epochs * (self.args.min_ratio[1] - self.args.min_ratio[0]) + self.args.min_ratio[0]
            else:
                min_ratio = epoch / self.args.epochs * (self.args.min_ratio[1] - self.args.min_ratio[0]) + self.args.min_ratio[0]
                # min_ratio = args.min_ratio[1]
                archs = sorted(archs, key=lambda x:x[1])

        for arch, score in archs:
            ratio = score if self.args.no_baseline else 1.
            out = self.model(self.data, arch)[self.train_idx]

            if self.args.min_clip > 0:
                ratio = max(ratio, self.args.min_clip)
            if self.args.max_clip > 0:
                ratio = min(ratio, self.args.max_clip)

            loss = F.nll_loss(out, self.data.y[self.train_idx], reduction="none") / self.args.repeat * ratio

            aggrement = (out.argmax(dim=1) == self.data.y[self.train_idx])
            cur_acc = aggrement.float().mean()
            if self.args.use_curriculum and (judgement is None or cur_acc > best_acc):
                # cal the judgement
                judgement = torch.ones_like(loss).float()
                bar = 1 / self.space.num_classes
                wrong_idxs = (~aggrement).nonzero()[:, 0] # .squeeze()
                # pass by the bar
                distributions = torch.exp(out)
                try:
                    wrong_idxs = wrong_idxs[distributions[wrong_idxs].max(dim=1)[0] > min(5 * bar, 0.7)]
                except:
                    import pdb
                    pdb.set_trace()
                sorted_idxs = distributions[wrong_idxs].max(dim=1)[0].sort(descending=True)[1][:int(self.args.max_ratio * out.size(0))]
                wrong_idxs = wrong_idxs[sorted_idxs]

                if min_ratio < 0:
                    judgement = judgement.bool()
                    judgement[wrong_idxs] = False
                else:
                    judgement[wrong_idxs] = min_ratio

                loss = loss.mean()
                best_acc = cur_acc
            else:
                if not self.args.use_curriculum:
                    loss = loss.mean()
                else:
                    if min_ratio < 0: loss = loss[judgement].mean()
                    else: loss = (loss * judgement).mean()

            loss.backward()
        optimizer.step()

        return loss.item(), cur_acc.item()

    # def _infer(self, mask="train"):
    #     if mask == "train":
    #         dataloader = self.train_loader
    #     elif mask == "val":
    #         dataloader = self.val_loader
    #     elif mask == "test":
    #         dataloader = self.test_loader
    #     metric, loss = self.estimator.infer(self.space, dataloader)
    #     return metric, loss

    def fit(self):
        optimizer = torch.optim.Adam(self.space.model.parameters(), lr=self.args.lr, weight_decay=self.args.wd)

        eta = self.args.eta
        best_performance = 0
        min_val_loss = float("inf")

        with trange(self.num_epochs, disable=self.disable_progress) as bar:
            for epoch in bar:
                """
                space training
                """
                self.space.train()
                eta = (
                    self.args.eta_max - self.args.eta
                ) * epoch / self.num_epochs + self.args.eta
                optimizer.zero_grad()

                train_loss, train_acc = self.train_graph(
                    optimizer,
                    epoch
                )


                """
                space evaluation
                """
                self.space.eval()
                # train_metric, train_loss = self._infer("train")
                # val_metric, val_loss = self._infer("val")
                # test_metric, test_loss = self._infer("test")

                # if min_val_loss > val_loss:
                #     min_val_loss, best_performance = val_loss, val_metric["auc"]
                #     self.space.keep_prediction()

                # bar.set_postfix(
                #     {
                #         "train_auc": train_metric["auc"],
                #         "val_auc": val_metric["auc"],
                #         # "test_auc": test_metric["auc"],
                #     }
                # )

        return best_performance, min_val_loss

    def search(self, space: BaseSpace, data, estimator: BaseEstimator):
        self.estimator = estimator
        self.space = space.to(self.device)
        self.prepare(data)
        perf, val_loss = self.fit()
        return space.parse_model(None)