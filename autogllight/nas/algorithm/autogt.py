# "AutoGT: Automated Graph Transformer Architecture Search" ICLR 23'


import json
import os
import random
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm, trange

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


    def train_supernet(self, optimizer, scheduler):
        self.space.train()
        total_loss = 0
        for batched_data in self.train_loader[1:]:
            optimizer.zero_grad()
            y_hat = self.space(batched_data, self.space.model.gen_params()).squeeze()
            y_gt = batched_data.y.view(-1)
            loss = F.binary_cross_entropy_with_logits(y_hat, y_gt.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.double() * batched_data.y.shape[0]
            scheduler.step()
        return total_loss / self.train_loader[0]


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

    def test(self, data_loader, params=None):
        self.space.eval()
        total_correct = 0
        total_loss = 0
        for batched_data in data_loader[1:]:
            out = self.space(batched_data, params).squeeze()
            total_correct += int(((out > 0.5) == batched_data.y).sum())
            loss = F.binary_cross_entropy_with_logits(out, batched_data.y.view(-1).float())
            total_loss += loss.double() * batched_data.y.shape[0]
        return total_loss / data_loader[0], total_correct / data_loader[0]


    def _infer(self, mask="train"):

        if mask == "train":
            mask = self.train_idx
        elif mask == "valid":
            mask = self.valid_idx
        else:
            mask = self.test_idx

        metric, loss = self.estimator.infer(self.space, self.data, self.args.arch, mask=mask)
        return metric, loss

    def gen_params(self, path, spa, edg, pma, cen):
        with open(path, 'r') as f:
            dic = json.load(f)
        depth = random.choice(dic['depth'])
        layers = []
        for _ in range(0, depth):
            layer = []
            hidden_in = random.choice(dic['hidden_in'])
            num_heads = random.choice(dic['num_heads'])
            att_size = random.choice(dic['att_size'])
            hidden_mid = random.choice(dic['hidden_mid'])
            ffn_size = random.choice(dic['ffn_size'])
            mask = random.choice(dic['mask'])
            layer.append((hidden_in, num_heads, att_size, hidden_mid, ffn_size, mask))
            cen = random.choice([True, False])
            eig = random.choice([True, False])
            svd = random.choice([True, False])
            layer.append((cen, eig, svd))
            spa = spa > 0
            edg = edg > 0
            pma = random.choice([True, False])
            layer.append((spa, edg, pma))
            layers.append(tuple(layer))
        return (depth, tuple(layers))

    def get_directory(self):
        directory = f'./PROTEINS/checkpoints/{self.args.dataset_name}_4/{str(self.args.seed)}/{str(self.args.data_split)}/'
        return directory

    def get_ord(self, params):
        spa = int(params[1][0][2][0])
        edg = int(params[1][0][2][1])
        pma = int(params[1][0][2][2])
        cen = int(params[1][0][1][0])
        ord = spa + (edg << 1)# + (pma << 2) + (cen << 3)
        return ord

    def gen_layer(self, dic, params):
        layer = []
        hidden_in = random.choice(dic['hidden_in'])
        num_heads = random.choice(dic['num_heads'])
        att_size = random.choice(dic['att_size'])
        hidden_mid = random.choice(dic['hidden_mid'])
        ffn_size = random.choice(dic['ffn_size'])
        mask = random.choice(dic['mask'])
        layer.append([hidden_in, num_heads, att_size, hidden_mid, ffn_size, mask])
        cen = random.choice([True, False])
        eig = random.choice([True, False])
        svd = random.choice([True, False])
        layer.append([cen, eig, svd])
        spa = params[1][0][2][0]
        edg = params[1][0][2][1]
        pma = random.choice([True, False])
        layer.append([spa, edg, pma])
        return layer

    def evolution(self, directory):
        start = time.time()
        with open(self.args.path, 'r') as f:
            dic = json.load(f)
        models = []
        for ord in range(4):
            sub_name = 'supernet_' + str(ord) + '.pt'
            models.append(self.space.load_model(directory + sub_name)[0])

        information = {}
        population = []
        candidates = []

        def is_legal(params):
            if params in information:
                return False
            info = {}
            ord = self.get_ord(params)
            _, valid_acc = self.test(models[ord], self.valid_loader, params)
            _, test_acc_ = self.test(models[ord], self.test_loader, params)
            info['valid_acc'] = valid_acc
            info['test_acc_'] = test_acc_
            print('Top-1 Valid Accuracy = {}, Top-1 Test Accuracy = {}, Parameters = {}'.format(
                valid_acc, test_acc_, params))
            information[params] = info
            return True

        def get_mutation():
            start = time.time()
            print('Start Mutation!')
            result = []

            def random_function():
                params = random.choice(population)
                depth, layers = params
                layers = [list(list(item) for item in layer) for layer in layers]
                if random.random() < self.args.m_prob:
                    new_depth = random.choice(dic['depth'])
                    if new_depth > depth:
                        layers = layers + [self.gen_layer(dic, params) for _ in range(new_depth - depth)]
                    else:
                        layers = layers[:new_depth]
                    depth = new_depth
                for i in range(depth):
                    if random.random() < self.args.m_prob:
                        layers[i][0][0] = random.choice(dic['hidden_in'])
                    if random.random() < self.args.m_prob:
                        layers[i][0][1] = random.choice(dic['num_heads'])
                    if random.random() < self.args.m_prob:
                        layers[i][0][2] = random.choice(dic['att_size'])
                    if random.random() < self.args.m_prob:
                        layers[i][0][3] = random.choice(dic['hidden_mid'])
                    if random.random() < self.args.m_prob:
                        layers[i][0][4] = random.choice(dic['ffn_size'])
                    if random.random() < self.args.m_prob:
                        layers[i][0][5] = random.choice(dic['mask'])
                    if random.random() < self.args.m_prob:
                        layers[i][1][0] = random.choice([True, False])
                    if random.random() < self.args.m_prob:
                        layers[i][1][1] = random.choice([True, False])
                    if random.random() < self.args.m_prob:
                        layers[i][1][2] = random.choice([True, False])
                if random.random() < self.args.m_prob:
                    flag = random.choice([True, False])
                    for i in range(depth):
                        layers[i][2][0] = flag
                if random.random() < self.args.m_prob:
                    flag = random.choice([True, False])
                    for i in range(depth):
                        layers[i][2][1] = flag
                if random.random() < self.args.m_prob:
                    flag = random.choice([True, False])
                    for i in range(depth):
                        layers[i][2][2] = flag

                layers = tuple([tuple([tuple(item) for item in layer]) for layer in layers])
                result = tuple([depth, layers])
                return result

            iters = self.args.mutation_num * 10
            while len(result) < self.args.mutation_num and iters > 0:
                iters -= 1
                params = random_function()
                if not is_legal(params):
                    continue
                result.append(params)

            end = time.time()
            print("End Mutation! Use time: {} s".format(end - start))
            return result

        def get_hybridization():
            start = time.time()
            print('Start Hybridization!')
            result = []

            def random_function():
                params1 = random.choice(population)
                params2 = random.choice(population)
                iters = self.args.population_num
                while params1[0] != params2[0] and iters > 0:
                    iters -= 1
                    params1 = random.choice(population)
                    params2 = random.choice(population)
                AT_choice = random.choice([params1[1][0][2], params2[1][0][2]])
                layers = []
                for i in range(params1[0]):
                    shape_choice = []
                    for j in range(6):
                        shape_choice.append(random.choice([params1[1][i][0][j], params2[1][i][0][j]]))
                    PE_choice = []
                    for j in range(3):
                        PE_choice.append(random.choice([params1[1][i][1][j], params2[1][i][1][j]]))
                    layers.append(tuple([tuple(shape_choice), tuple(PE_choice), AT_choice]))
                params = (params1[0], tuple(layers))
                return params

            iters = 10 * self.args.hybridization_num
            while len(result) < self.args.hybridization_num and iters > 0:
                iters -= 1
                params = random_function()
                if not is_legal(params):
                    continue
                result.append(params)

            end = time.time()
            print("End Hybridization! Use time: {} s".format(end - start))
            return result

        # epoch = 0
        # while epoch < self.args.evol_epochs:
        for epoch in tqdm(range(self.args.evol_epochs)):
            while len(candidates) < self.args.population_num:
                spa = random.choice([True, False])
                edg = random.choice([True, False])
                pma = random.choice([True, False])
                cen = random.choice([True, False])
                params = self.gen_params(self.args.path, spa, edg, pma, cen)
                if not is_legal(params):
                    continue
                candidates.append(params)

            population += candidates
            population.sort(key=lambda x: information[x]['valid_acc'], reverse=True)
            population = population[:self.args.population_num]

            print('epoch = {} : top {} result'.format(epoch, len(population)))
            for i, params in enumerate(population):
                print('No.{} Top-1 Valid Accuracy = {}, Top-1 Test Accuracy = {}, Parameters = {}'.format(
                    i + 1, information[params]['valid_acc'], information[params]['test_acc_'], params))

            # epoch += 1
            if epoch != self.args.evol_epochs:
                candidates = get_mutation() + get_hybridization()

        end = time.time()
        print("Evolution Ended! Use time: {} s".format(end - start))



    def fit(self):
        optimizer, lr_scheduler = self.space.model.configure_optimizers()
        scheduler = lr_scheduler['scheduler']
        best_performance = 0
        min_val_loss = float("inf")

        for epoch in tqdm(range(self.args.split_epochs)):
            self.train_supernet(optimizer, scheduler)

        # save model
        directory = self.get_directory()
        # print('dir: ', directory)
        # exit()
        if not os.path.exists(directory):
            os.makedirs(directory)
        name = 'supernet.pt'
        self.space.save_model(optimizer, scheduler, directory + name)


        # load model and split
        for ord in range(4):
            spa = int((ord & 1) != 0)
            edg = int((ord & 2) != 0)
            pma = int((ord & 4) != 0)
            cen = int((ord & 8) != 0)
            _, optimizer, scheduler = self.space.load_model(directory + name)
            for epoch in tqdm(range(self.args.split_epochs, self.args.end_epochs)):
                params = self.gen_params(self.args.path, spa, edg, pma, cen)
                self.train(optimizer, scheduler, params)
            sub_name = 'supernet_' + str(ord) + '.pt'
            self.space.save_model(optimizer, scheduler, directory + sub_name)

        # evolution
        self.evolution(directory)


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

        #         train_loss, train_acc = self.test(model, train_loader)
        #         valid_loss, valid_acc = self.test(model, valid_loader)
        #         test__loss, test_acc_ = self.test(model, test_loader_)
        #         print("Epoch {: >3}: Train Loss: {:.3f}, Train Acc: {:.2%}, Valid Loss: {:.3f}, Valid Acc: {:.2%}, Test Loss: {:.3f}, Test Acc: {:.2%}".format(epoch, loss, train_acc, valid_loss, valid_acc, test__loss, test_acc_))
        #         if valid_acc > best_valid:
        #             best_valid = valid_acc
        #             best_test_ = test_acc_
        #             worst_test = test_acc_
        #         elif valid_acc == best_valid:
        #             if test_acc_ > best_test_:
        #                 best_test_ = test_acc_
        #             elif test_acc_ < worst_test:
        #                 worst_test = test_acc_
        #         results[epoch, 0] = train_acc
        #         results[epoch, 1] = valid_acc
        #         results[epoch, 2] = test_acc_



        #         """
        #         space evaluation
        #         """
        #         # self.space.eval()
        #         # train_acc, _ = self._infer("train")
        #         # val_acc, val_loss = self._infer("val")

        #         # if min_val_loss > val_loss:
        #         #     min_val_loss, best_performance = val_loss, val_acc
        #         #     self.space.keep_prediction()

        #         bar.set_postfix(train_acc=train_acc["acc"], val_acc=val_acc["acc"])

        # return best_performance, min_val_loss

    def search(self, space: BaseSpace, data, estimator: BaseEstimator):
        self.estimator = estimator
        self.space = space.to(self.device)
        self.prepare(data)
        # perf, val_loss = self.fit()
        self.fit()
        return space.parse_model(None)