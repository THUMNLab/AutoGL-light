# "AutoGT: Automated Graph Transformer Architecture Search" ICLR 23'


import random
import os

import torch
import torch.nn.functional as F
from tqdm import trange

from ..estimator.base import BaseEstimator
from ..space import BaseSpace
from .base import BaseNAS


class Autogt(BaseNAS):
    """
    AutoGT trainer.

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
        self.train_loader = data[0]
        self.valid_loader = data[1]
        self.test_loader = data[2]


    def train(self, optimizer, scheduler, params=None):
        self.space.train()
        total_loss = 0
        for batched_data in self.train_loader[1:]:
            optimizer.zero_grad()
            y_hat = self.space(batched_data, params).squeeze()
            y_gt = batched_data.y.view(-1)
            loss = F.binary_cross_entropy_with_logits(y_hat, y_gt.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.double() * batched_data.y.shape[0]
            scheduler.step()
        return total_loss / self.train_loader[0]

    def _infer(self, mask="train"):

        if mask == "train":
            mask = self.train_idx
        elif mask == "valid":
            mask = self.valid_idx
        else:
            mask = self.test_idx

        metric, loss = self.estimator.infer(self.space, self.data, self.args.arch, mask=mask)
        return metric, loss

    def get_directory(self):
        directory = f'./PROTEINS/checkpoints/{self.args.dataset_name}_4/{str(self.args.seed)}/{str(self.args.data_split)}/'
        return directory

    def load_model(self,

    def save_model(self, optimizer, scheduler, path):
        print("Saving Model to Path: " + path)
        torch.save({'model': self.space.model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}, path)
        print("Save Successfully!")

    def fit(self):
        optimizer, lr_scheduler = self.space.model.configure_optimizers()
        scheduler = lr_scheduler['scheduler']
        best_performance = 0
        min_val_loss = float("inf")

        for epoch in range(self.args.split_epochs):
            self.train_supernet(optimizer, scheduler)

        # save model
        directory = self.get_directory()
        if not os.path.exists(directory):
            os.makedirs(directory)
        name = 'supernet.pt'
        self.save_model(optimizer, scheduler, directory + name)


        # load model and split
        for ord in range(4):
            spa = int((ord & 1) != 0)
            edg = int((ord & 2) != 0)
            pma = int((ord & 4) != 0)
            cen = int((ord & 8) != 0)
            model, optimizer, scheduler = load_model(args, directory + name)
            for epoch in range(args.split_epochs, args.end_epochs):
                params = gen_params(args.path, spa, edg, pma, cen)
                train(model, train_loader, optimizer, scheduler, params)
            sub_name = 'supernet_' + str(ord) + '.pt'
            save_model(model, optimizer, scheduler, directory + sub_name)



        # with trange(self.num_epochs, disable=self.disable_progress) as bar:
        #     for epoch in bar:
        #         """
        #         space training
        #         """

        #         loss = self.train(optimizer, scheduler)



        #         """
        #         space evaluation
        #         """
        #         continue
        #         self.space.eval()
        continue

                train_loss, train_acc = self.test(model, train_loader)
                valid_loss, valid_acc = self.test(model, valid_loader)
                test__loss, test_acc_ = self.test(model, test_loader_)
                print("Epoch {: >3}: Train Loss: {:.3f}, Train Acc: {:.2%}, Valid Loss: {:.3f}, Valid Acc: {:.2%}, Test Loss: {:.3f}, Test Acc: {:.2%}".format(epoch, loss, train_acc, valid_loss, valid_acc, test__loss, test_acc_))
                if valid_acc > best_valid:
                    best_valid = valid_acc
                    best_test_ = test_acc_
                    worst_test = test_acc_
                elif valid_acc == best_valid:
                    if test_acc_ > best_test_:
                        best_test_ = test_acc_
                    elif test_acc_ < worst_test:
                        worst_test = test_acc_
                results[epoch, 0] = train_acc
                results[epoch, 1] = valid_acc
                results[epoch, 2] = test_acc_



                """
                space evaluation
                """
                # self.space.eval()
                # train_acc, _ = self._infer("train")
                # val_acc, val_loss = self._infer("val")

                # if min_val_loss > val_loss:
                #     min_val_loss, best_performance = val_loss, val_acc
                #     self.space.keep_prediction()

                bar.set_postfix(train_acc=train_acc["acc"], val_acc=val_acc["acc"])

        return best_performance, min_val_loss

    def search(self, space: BaseSpace, data, estimator: BaseEstimator):
        self.estimator = estimator
        self.space = space.to(self.device)
        self.prepare(data)
        perf, val_loss = self.fit()
        return space.parse_model(None)