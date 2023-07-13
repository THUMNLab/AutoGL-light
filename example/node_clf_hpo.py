import numpy as np
import typing as _typing

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

from torch_geometric.nn import GCNConv
from .base import EarlyStopping
from autogllight.hpo import build_hpo_from_name, LogRangeHP

# pyg version

class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

class Trainer:
    def __init__(self, dataset, model, init, lr, weight_decay, max_epoch, early_stopping_round, device):
        super().__init__()
        self.device = device
        self.data = dataset[0]
        self.data = self.data.to(self.device)
        self.model = model
        self.init = init
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epoch = max_epoch
        self.early_stopping_round = early_stopping_round
        self.early_stopping = EarlyStopping(
            patience=early_stopping_round, verbose=False
        )

        if init is True:
            self.initialize()

    def set_hp(self, hps: dict):
        for i in hps:
            setattr(self, i, hps[i])

    def _initialize(self):
        self.encoder.initialize()
        if self.decoder is not None:
            self.decoder.initialize(self.encoder)

    def train(self):
        """
        Train on the given dataset.
        Parameters
        ----------
        model: nn.Module
        -------
        """
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), self.lr, weight_decay=self.weight_decay)
        for epoch in range(1, self.max_epoch + 1):
            train_mask = self.data.train_mask
            pred = self.model(self.data)
            loss = F.cross_entropy(pred[train_mask], self.data.y[train_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if self.data.val_mask is not None:
                val_mask = self.data.val_mask
            else:
                val_mask = None

            if val_mask is not None:
                val_loss = F.cross_entropy(pred[val_mask], self.data.y[val_mask])
                self.early_stopping(val_loss, self.model)
                if self.early_stopping.early_stop:
                    LOGGER.debug("Early stopping at %d", epoch)
                    break

    @torch.no_grad()
    def evaluate(self):
        """
        Parameters
        ----------
        model: nn.Module
        """

        mask = getattr(self.data, 'test_mask')
        self.model.eval()
        pred = self.model(self.data)
        acc = (pred.argmax(1) == self.data.y)[mask].float().mean()
        return acc.item()

def do_train_without_hpo():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from torch_geometric.datasets import Planetoid
    dataset = Planetoid(root='~/data/', name='Cora')

    num_features = dataset[0].x.size(1)
    num_classes = dataset[0].y.max().item() + 1
    model = GCN(num_features, num_classes).to(device)

    node_trainer = Trainer(
        dataset=dataset,
        model=model,
        init=False,
        lr=1e-2,
        weight_decay=5e-4,
        max_epoch=200,
        early_stopping_round=200,
        device=device,
    )

    node_trainer.train()
    result = node_trainer.evaluate()
    print("Acc:", result)

def do_hpo():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from torch_geometric.datasets import Planetoid
    dataset = Planetoid(root='~/data/', name='Cora')

    num_features = dataset[0].x.size(1)
    num_classes = dataset[0].y.max().item() + 1


    def f(hps):
        model = GCN(num_features, num_classes).to(device)
        trainer = Trainer(
            dataset=dataset,
            model=model,
            init=False,
            lr=1e-2,
            weight_decay=5e-4,
            max_epoch=200,
            early_stopping_round=200,
            device=device,
        )
        trainer.set_hp(hps)
        trainer.train()
        result = trainer.evaluate()
        print("acc", result)
        # high is better
        return -result

    hp_space = [
        LogRangeHP("lr", 1e-4, 1e-1),
        LogRangeHP("weight_decay", 1e-4, 1e-2),
    ]
    hpo = build_hpo_from_name("tpe", hp_space, f)
    print(hpo.optimize())

do_hpo()