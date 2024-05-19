# "A surrogate evolutionary neural architecture search algorithm for graph neural networks" Applied Soft Computing 144 (2023)

import torch
import torch.nn.functional as F
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import numpy as np
import scipy.stats as stats
from tqdm import trange

from ..space import BaseSpace, CTFSpace
from .base import BaseNAS


class CTF(BaseNAS):
    def __init__(
        self,
        run_times=5,
        max_gen_num=10,
        num_epochs=10,
        lr=0.005,
        wd=0.0005,
        device="auto",
    ) -> None:
        super().__init__(device=device)
        self.run_times = run_times
        self.max_gen_num = max_gen_num

        self.num_epochs = num_epochs
        self.lr = lr
        self.wd = wd

        # id of evaluated individuals -> Selection
        self.eval_indv_list = []
        self.eval_genoBin_sim_rec = {}
        # [genoBin_sim, val_loss, test_acc] of evaluated individuals
        self.eval_genoBin_sim_loss_acc_list = []
        # [genoDec, val loss] of evaluated individuals -> Surrogate Model
        self.eval_genoDec_loss_list = []
        # predictor accuracy
        self.pred_acc_list = []

    def train_graph(self, genotype, tuning_params=None):
        if tuning_params is None:
            model = self.space.build_graph(genotype).to(self.device)
        else:
            model = self.space.tune_graph(genotype, tuning_params).to(self.device)

        criterion = torch.nn.NLLLoss().to(self.device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=self.wd
        )

        min_val_loss, patience, best_test_acc = float("inf"), 0, 0

        for _ in range(self.num_epochs):
            model.train()
            optimizer.zero_grad()

            out = model(
                self.dataset.x.to(self.device), self.dataset.edge_index.to(self.device)
            )
            out = F.log_softmax(out, 1)
            loss = criterion(
                out[self.dataset.train_mask],
                self.dataset.y[self.dataset.train_mask].to(self.device),
            )
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                model.eval()
                out = model(
                    self.dataset.x.to(self.device),
                    self.dataset.edge_index.to(self.device),
                )
                out = F.log_softmax(out, 1)
                loss_val = criterion(
                    out[self.dataset.val_mask],
                    self.dataset.y[self.dataset.val_mask].to(self.device),
                )
                loss_val = loss_val.item()

                predict = (
                    out[self.dataset.test_mask.to(self.device)]
                    .max(1)[1]
                    .type_as(
                        self.dataset.y.to(self.device)[
                            self.dataset.test_mask.to(self.device)
                        ]
                    )
                )

                correct = predict.eq(
                    self.dataset.y.to(self.device)[
                        self.dataset.test_mask.to(self.device)
                    ]
                ).double()
                acc_test = correct.sum() / len(
                    self.dataset.y.to(self.device)[
                        self.dataset.test_mask.to(self.device)
                    ]
                )
                acc_test = acc_test.item()

                if loss_val < min_val_loss:
                    min_val_loss = loss_val
                    patience = 0
                    best_test_acc = acc_test
                elif patience < 150:
                    patience += 1
                else:
                    break

        return round(min_val_loss, 4), round(best_test_acc, 4)

    def _init_search_space(self):
        for indv in self.space.population:
            val_loss, test_acc = self.train_graph(indv.genoBin_sim)

            indv.val_loss = val_loss
            indv.test_acc = test_acc

            self.eval_genoBin_sim_rec[str(indv.genoBin_sim)] = len(self.eval_indv_list)

            self.eval_indv_list.append(indv)
            self.eval_genoBin_sim_loss_acc_list.append(
                [indv.genoBin_sim, val_loss, test_acc]
            )
            self.eval_genoDec_loss_list.append([indv.genoDec, val_loss])
        self.space.to_be_eval = False

    def _predict(self):
        inputs = np.array([x[0] for x in self.eval_genoDec_loss_list])
        targets = np.array([x[1] for x in self.eval_genoDec_loss_list])

        mlp_predictor = MLPRegressor(random_state=1, max_iter=5000)
        mlp_predictor.fit(inputs, targets)
        mlp_prediction = mlp_predictor.predict(inputs)
        mlp_rho, _ = stats.spearmanr(mlp_prediction, targets)

        gbdt_predictor = GradientBoostingRegressor()
        gbdt_predictor.fit(inputs, targets)
        gbdt_prediction = gbdt_predictor.predict(inputs)
        gbdt_rho, _ = stats.spearmanr(gbdt_prediction, targets)

        rf_predictor = RandomForestRegressor()
        rf_predictor.fit(inputs, targets)
        rf_prediction = rf_predictor.predict(inputs)
        rf_rho, _ = stats.spearmanr(rf_prediction, targets)

        if mlp_rho >= gbdt_rho and mlp_rho >= rf_rho:
            predictor = mlp_predictor
            prediction = mlp_prediction
        elif gbdt_rho >= mlp_rho and gbdt_rho >= rf_rho:
            predictor = gbdt_predictor
            prediction = gbdt_prediction
        else:
            predictor = rf_predictor
            prediction = rf_prediction

        # label all
        for indv in self.space.offspring:
            genoBin_sim_str = str(indv.genoBin_sim)
            if genoBin_sim_str in self.eval_genoBin_sim_rec:
                idx = self.eval_genoBin_sim_rec[genoBin_sim_str]
                indv.val_loss = self.eval_genoBin_sim_loss_acc_list[idx][1]
            else:
                indv.val_loss = predictor.predict(np.array([indv.genoDec])).item()

        # select
        selected_index = self.space._select_index()
        for index in selected_index:
            indv = self.space.offspring[index]
            genoBin_sim_str = str(indv.genoBin_sim)

            if genoBin_sim_str in self.eval_genoBin_sim_rec:
                idx = self.eval_genoBin_sim_rec[genoBin_sim_str]
                indv.val_loss = self.eval_genoBin_sim_loss_acc_list[idx][1]
            else:
                prediction = np.concatenate((prediction, [indv.val_loss]))
                indv.val_loss, indv.test_acc = self.train_graph(indv.genoBin_sim)

                targets = np.concatenate((targets, [indv.val_loss]))

                self.eval_genoBin_sim_rec[genoBin_sim_str] = len(self.eval_indv_list)

                self.eval_indv_list.append(indv)
                self.eval_genoBin_sim_loss_acc_list.append(
                    [indv.genoBin_sim, indv.val_loss, indv.test_acc]
                )
                self.eval_genoDec_loss_list.append([indv.genoDec, indv.val_loss])

        rho, _ = stats.spearmanr(prediction, targets)
        self.pred_acc_list.append(rho)

    def fit(self):
        # init search space
        self._init_search_space()

        for t in trange(self.run_times, desc="Searching"):
            for gen in trange(self.max_gen_num, desc="Generation"):
                # space training
                self.space.train()

                # space evaluation
                self._predict()

                # space forward
                self.space.forward()

    def search(self, space: CTFSpace, dataset, estimator, return_model=True):
        self.estimator = estimator  # TODO: use autogllight.estimator
        self.space = space.to(self.device)
        self.dataset = dataset

        if not self.space.to_be_eval:
            self.space.instantiate()
        self.fit()

    def _fine_tune(self, genoBin):
        # return self.space.tune_graph(genoBin)
        return self.space.build_graph(genoBin)

    def get_selection(self):
        min_loss = float("inf")
        min_loss_idx = -1
        for idx, indv in enumerate(self.eval_indv_list):
            if indv.val_loss < min_loss:
                min_loss = indv.val_loss
                min_loss_idx = idx

        best_indv = self.eval_indv_list[min_loss_idx]
        best_val_loss = best_indv.val_loss
        best_test_acc = best_indv.test_acc
        best_genoBin_sim = best_indv.genoBin_sim

        selection = self._fine_tune(best_genoBin_sim)

        print(f"Best validation loss: {best_val_loss}")
        print(f"Best test accuracy: {best_test_acc}")
        return selection
