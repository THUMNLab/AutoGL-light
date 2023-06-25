import torch
from torch import nn
import torch.nn.functional as F


def act_map(act):
    if act == "linear":
        return lambda x: x
    elif act == "elu":
        return F.elu
    elif act == "sigmoid":
        return torch.sigmoid
    elif act == "tanh":
        return torch.tanh
    elif act == "relu":
        return torch.nn.functional.relu
    elif act == "relu6":
        return torch.nn.functional.relu6
    elif act == "softplus":
        return torch.nn.functional.softplus
    elif act == "leaky_relu":
        return torch.nn.functional.leaky_relu
    else:
        raise Exception("wrong activate function")


class LambdaModule(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.lambd)


class StrModule(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.str = lambd

    def forward(self, *args, **kwargs):
        return self.str

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.str)


def act_map_nn(act):
    return LambdaModule(act_map(act))


def map_nn(l):
    return [StrModule(x) for x in l]


from autogllight.utils.backend import is_dgl

if is_dgl():
    from .operation_dgl import gnn_map, GeoLayer
else:
    from .operation_pyg import gnn_map, GeoLayer
