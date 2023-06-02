from collections import namedtuple
from torch.nn import Module
import torch.nn as nn
import torch

Genotype = namedtuple("Genotype", "normal normal_concat")
Genotype_normal = namedtuple("Genotype_normal", "normal normal_concat")


act_list = ["sigmoid", "tanh", "relu", "linear", "elu"]


class LinearConv(Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(LinearConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear = torch.nn.Linear(in_channels, out_channels, bias)

    def forward(self, x, edge_index, edge_weight=None):
        return self.linear(x)

    def __repr__(self):
        return "{}({}, {})".format(
            self.__class__.__name__, self.in_channels, self.out_channels
        )


class SkipConv(Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(SkipConv, self).__init__()
        self.out_dim = out_channels

    def forward(self, x, edge_index, edge_weight=None):
        return x

    def __repr__(self):
        return "{}({}, {})".format(
            self.__class__.__name__, self.in_channels, self.out_channels
        )


class ZeroConv(Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(ZeroConv, self).__init__()
        self.out_dim = out_channels

    def forward(self, x, edge_index, edge_weight=None):
        return torch.zeros([x.size(0), self.out_dim]).to(x.device)

    def __repr__(self):
        return "{}({}, {})".format(
            self.__class__.__name__, self.in_channels, self.out_channels
        )
