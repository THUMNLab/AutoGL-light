# "Graph Neural Architecture Search Under Distribution Shifts" ICML 22'

import logging
from itertools import cycle

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

    def _infer(self, model: BaseSpace, dataset, estimator: BaseEstimator, mask="train"):
        metric, loss = estimator.infer(model, dataset, mask=mask, adjs=self.adjs)
        return metric, loss

    def train_graph(
        self,
        data,
        model,
        criterion,
        model_optimizer,
        arch_optimizer,
        gnn0_optimizer,
        eta,
    ):
        model.train()
        total_loss = 0
        accuracy = 0
        # data:[dataset, train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader
        train_iters = data[4].__len__() // self.args.w_update_epoch + 1
        print(
            "train_iters:{},train_data_num:{}".format(
                train_iters, range(train_iters * self.args.w_update_epoch)
            )
        )

        zip_valid_data = list(zip(range(train_iters), cycle(data[5])))

        for id, train_data in enumerate(data[4]):
            model_optimizer.zero_grad()
            arch_optimizer.zero_grad()
            gnn0_optimizer.zero_grad()
            train_data = train_data.to(self.device)

            if id % 15 == 5:
                model.ag.set = "train"
            else:
                model.ag.set = "nooutput"
            output0, output, cosloss, sslout = model(train_data)
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
            total_loss += my_loss.item()

            my_loss.backward()
            model_optimizer.step()
            gnn0_optimizer.step()

            if self.args.alpha_mode == "train_loss":
                arch_optimizer.step()

            if self.args.alpha_mode == "valid_loss":
                valid_data = zip_valid_data[iter][1].to(self.device)
                model_optimizer.zero_grad()
                arch_optimizer.zero_grad()
                output = model(valid_data)
                output = output.to(self.device)

                error_loss = criterion(output, valid_data.y.view(-1))
                error_loss.backward()
                arch_optimizer.step()

        return accuracy / len(data[4].dataset), total_loss / len(data[4].dataset)

    def infer_graph_ogbg(self, data_, model, evaluator):
        model.eval()

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
                        pred0, pred, _, _ = model(batch)

                    y_true.append(batch.y.view(pred.shape).detach().cpu())
                    y_pred0.append(pred0.detach().cpu())
                    y_pred.append(pred.detach().cpu())

            y_true = torch.cat(y_true, dim=0).numpy()
            y_pred0 = torch.cat(y_pred0, dim=0).numpy()
            y_pred = torch.cat(y_pred, dim=0).numpy()

            input_dict = {"y_true": y_true, "y_pred": y_pred0}
            perf0 = evaluator.eval(input_dict)
            input_dict = {"y_true": y_true, "y_pred": y_pred}
            # print(y_pred)
            # print(y_true)
            perf = evaluator.eval(input_dict)
            return perf0, perf

        p0, p = evaluate(data_[4])
        train_perf0, train_perf = p0["rocauc"], p["rocauc"]
        p0, p = evaluate(data_[5])
        val_perf0, val_perf = p0["rocauc"], p["rocauc"]
        p0, p = evaluate(data_[6])
        test_perf0, test_perf = p0["rocauc"], p["rocauc"]
        return train_perf0, train_perf, val_perf0, val_perf, test_perf0, test_perf

    def get_valid_loss(self, data, model, criterion):
        model.train()
        total_loss = 0
        accuracy = 0
        train_iters = data[4].__len__() // self.args.w_update_epoch + 1

        for train_data in data[5]:
            train_data = train_data.to(self.device)

            model.ag.set = "valid"
            output0, output, cosloss, sslout = model(train_data)
            output = output.to(self.device)

            error_loss = criterion(
                output.to(torch.float32), train_data.y.to(torch.float32)
            )

            total_loss += error_loss.item()

        return total_loss / len(data[5].dataset)

    def fit(self, data):
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
                """space training"""
                self.space.train()

                eta = (
                    self.args.eta_max - self.args.eta
                ) * epoch / self.num_epochs + self.args.eta

                optimizer.zero_grad()
                arch_optimizer.zero_grad()
                gnn0_optimizer.zero_grad()

                train_acc, train_obj = self.train_graph(
                    data,
                    self.space,
                    criterion,
                    optimizer,
                    arch_optimizer,
                    gnn0_optimizer,
                    eta,
                )
                scheduler_arch.step()
                scheduler_gnn0.step()

                """space evaluation"""
                self.space.eval()

                (
                    train_acc0,
                    train_acc,
                    valid_acc0,
                    valid_acc,
                    test_acc0,
                    test_acc,
                ) = self.infer_graph_ogbg(data, self.space, self.estimator)
                valid_loss = self.get_valid_loss(data, self.space, criterion)

                if min_val_loss > valid_loss:
                    min_val_loss, best_test = valid_loss, test_acc
                    patience = 0
                else:
                    patience += 1
                    if patience == 1000:
                        break

                if epoch % 1 == 0:
                    logging.info(
                        "epoch=%s, train_acc=%f, train_loss=%f, valid_acc=%f, test_acc=%f, explore_num=%s",
                        epoch,
                        train_acc,
                        train_obj,
                        valid_acc,
                        test_acc,
                        self.space.explore_num,
                    )
                    print(
                        "epochACC0={}, train_acc={:.04f}, train_loss={:.04f},valid_acc={:.04f},test_acc={:.04f},explore_num={}".format(
                            epoch,
                            train_acc0,
                            train_obj,
                            valid_acc0,
                            test_acc0,
                            self.space.explore_num,
                        )
                    )
                    print(
                        "epoch={}, train_acc={:.04f}, train_loss={:.04f},valid_acc={:.04f},test_acc={:.04f},explore_num={}".format(
                            epoch,
                            train_acc,
                            train_obj,
                            valid_acc,
                            test_acc,
                            self.space.explore_num,
                        )
                    )

                # train_acc, _ = self._infer(self.space, data, self.estimator, "train")
                # val_acc, val_loss = self._infer(self.space, data, self.estimator, "val")
                # if val_loss < min_val_loss:
                #     min_val_loss = val_loss
                #     best_performance = val_acc
                #     # self.space.keep_prediction()

                # bar.set_postfix(train_acc=train_acc["acc"], val_acc=val_acc["acc"])
                bar.set_postfix(train_acc=train_acc, val_acc=valid_loss)
                # print("acc:" + str(train_acc) + " val_acc" + str(val_acc))

        return best_performance, min_val_loss

    def search(self, space: BaseSpace, dataset, estimator):
        self.estimator = estimator
        self.space = space.to(self.device)
        perf, val_loss = self.fit(dataset)
        return space.parse_model(None)
