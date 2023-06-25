import argparse
import os.path as osp

import torch
import torch.nn.functional as F

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import MLP, GINConv, global_add_pool


class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            mlp = MLP([in_channels, hidden_channels, hidden_channels])
            self.convs.append(GINConv(nn=mlp, train_eps=False))
            in_channels = hidden_channels

        self.mlp = MLP([hidden_channels, hidden_channels, out_channels],
                       norm=None, dropout=0.5)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        x = global_add_pool(x, batch)
        return self.mlp(x)


class Trainer:
    def __init__(self, model, train_loader, test_loader, args):
        self.args = args
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    def train(self):
        for epoch in range(1, self.args.epochs + 1):
            loss = self._train(self._model, self.train_loader, self.optimizer)
            train_acc = self._test(self._model, self.train_loader)
            test_acc = self._test(self._model, self.test_loader)

    def _train(self, model, loader, optimizer):
        model.train()

        total_loss = 0
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = F.cross_entropy(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * data.num_graphs
        return total_loss / len(train_loader.dataset)


    @torch.no_grad()
    def _test(self, model, loader):
        model.eval()

        total_correct = 0
        for data in loader:
            data = data.to(device)
            pred = model(data.x, data.edge_index, data.batch).argmax(dim=-1)
            total_correct += int((pred == data.y).sum())
        return total_correct / len(loader.dataset)
