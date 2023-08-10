# "Graph Neural Architecture Search Under Distribution Shifts" ICML 22'

import logging

import torch
from tqdm import trange

from ..estimator.base import BaseEstimator
from ..space import BaseSpace
from .base import BaseNAS


class Graces(BaseNAS):
    """
    GRACES trainer.

    Parameters
    ----------
    num_epochs : int
        Number of epochs planned for training.
    device : str or torch.device
        The device of the whole process
    """

    def __init__(
        self,
        num_epochs=250,
        device="auto",
        disable_progress=False,
        args=None,
    ):
        super().__init__(device=device)
        self.num_epochs = num_epochs
        self.disable_progress = disable_progress
        self.args = args

    def train_graph(
        self,
        criterion,
        model_optimizer,
        arch_optimizer,
        gnn0_optimizer,
        eta,
    ):
        self.space.train()

        for id, train_data in enumerate(self.train_loader):
            model_optimizer.zero_grad()
            arch_optimizer.zero_grad()
            gnn0_optimizer.zero_grad()
            train_data = train_data.to(self.device)

            if id % 15 == 5:
                self.space.ag.set = "train"
            else:
                self.space.ag.set = "nooutput"
            output0, output, cosloss, sslout = self.space(train_data)
            output0 = output0.to(self.device)
            output = output.to(self.device)

            is_labeled = train_data.y == train_data.y
            error_loss0 = criterion(
                output0.to(torch.float32)[is_labeled],
                train_data.y.to(torch.float32)[is_labeled],
            )
            error_loss = criterion(
                output.to(torch.float32)[is_labeled],
                train_data.y.to(torch.float32)[is_labeled],
            )

            ssltarget = train_data.deratio.view(-1, 3)
            ssllossfun = torch.nn.L1Loss()
            sslloss = ssllossfun(sslout, ssltarget)

            my_loss = (1 - eta) * (
                error_loss0 + self.args.gamma * sslloss + self.args.beta * cosloss
            ) + eta * error_loss

            my_loss.backward()
            model_optimizer.step()
            gnn0_optimizer.step()
            arch_optimizer.step()

    def get_valid_loss(self, criterion):
        self.space.train()
        total_loss = 0
        accuracy = 0
        for train_data in self.val_loader:
            train_data = train_data.to(self.device)

            self.space.ag.set = "valid"
            output0, output, cosloss, sslout = self.space(train_data)
            output = output.to(self.device)

            error_loss = criterion(
                output.to(torch.float32), train_data.y.to(torch.float32)
            )

            total_loss += error_loss.item()

        return total_loss / len(self.val_loader.dataset)

    # gasso
    # def _infer(self, model: BaseSpace, dataset, estimator: BaseEstimator, mask="train"):
    #     metric, loss = estimator.infer(model, dataset, mask=mask, adjs=self.adjs)
    #     return metric, loss

    def _infer(self, mask="train"):
        self.space.eval()

        def evaluate(loader):
            y_true = []
            y_pred0 = []
            y_pred = []

            for step, batch in enumerate(loader):
                batch = batch.to(self.device)

                if batch.x.shape[0] == 1:
                    pass
                else:
                    with torch.no_grad():
                        pred0, pred, _, _ = self.space(batch)

                    y_true.append(batch.y.view(pred.shape).detach().cpu())
                    y_pred0.append(pred0.detach().cpu())
                    y_pred.append(pred.detach().cpu())

            y_true = torch.cat(y_true, dim=0).numpy()
            y_pred0 = torch.cat(y_pred0, dim=0).numpy()
            y_pred = torch.cat(y_pred, dim=0).numpy()

            input_dict = {"y_true": y_true, "y_pred": y_pred0}
            perf0 = self.estimator.eval(input_dict)
            input_dict = {"y_true": y_true, "y_pred": y_pred}
            perf = self.estimator.eval(input_dict)
            return perf0, perf

        if mask == "train":
            dataloader = self.train_loader
        elif mask == "val":
            dataloader = self.val_loader
        elif mask == "test":
            dataloader = self.test_loader

        p0, p = evaluate(dataloader)
        perf0, perf = p0["rocauc"], p["rocauc"]
        return perf0, perf

    def prepare(self, data):
        """
        data : list of data objects.
            [dataset, train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader]
        """
        self.train_loader = data[4]
        self.val_loader = data[5]
        self.test_loader = data[6]

    def fit(self):
        optimizer = torch.optim.Adam(
            self.space.supernet.parameters(),
            self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        arch_optimizer = torch.optim.Adam(
            self.space.ag.parameters(),
            self.args.arch_learning_rate,
            weight_decay=self.args.arch_weight_decay,
        )
        gnn0_optimizer = torch.optim.Adam(
            self.space.supernet0.parameters(),
            self.args.gnn0_learning_rate,
            weight_decay=self.args.gnn0_weight_decay,
        )
        scheduler_arch = torch.optim.lr_scheduler.CosineAnnealingLR(
            arch_optimizer,
            float(self.num_epochs),
            eta_min=self.args.arch_learning_rate_min,
        )
        scheduler_gnn0 = torch.optim.lr_scheduler.CosineAnnealingLR(
            gnn0_optimizer,
            float(self.num_epochs),
            eta_min=self.args.gnn0_learning_rate_min,
        )

        criterion = torch.nn.BCEWithLogitsLoss()

        eta = self.args.eta

        # Train model
        best_performance = 0
        min_val_loss = float("inf")
        min_train_loss = float("inf")

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
                arch_optimizer.zero_grad()
                gnn0_optimizer.zero_grad()

                self.train_graph(
                    criterion,
                    optimizer,
                    arch_optimizer,
                    gnn0_optimizer,
                    eta,
                )
                scheduler_arch.step()
                scheduler_gnn0.step()

                """
                space evaluation
                """
                self.space.eval()

                train_acc0, train_auc = self._infer("train")
                valid_acc0, valid_auc = self._infer("val")
                test_acc0, test_auc = self._infer("test")

                valid_loss = self.get_valid_loss(criterion)

                if min_val_loss > valid_loss:
                    min_val_loss, best_test = valid_loss, test_auc
                    self.space.keep_prediction()

                # bar.set_postfix(train_acc=train_acc, val_acc=valid_acc)
                bar.set_postfix(
                    {"train_acc": train_auc, "val_auc": valid_auc, "test_auc": test_auc}
                )
                # bar.set_postfix(train_auc=train_auc, val_auc=valid_auc)
                # print("acc:" + str(train_acc) + " val_acc" + str(val_acc))

        return best_performance, min_val_loss

    def search(self, space: BaseSpace, dataset, estimator):
        self.estimator = estimator
        self.space = space.to(self.device)
        self.prepare(dataset)
        perf, val_loss = self.fit()
        return space.parse_model(None)
