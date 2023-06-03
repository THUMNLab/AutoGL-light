# codes in this file are reproduced from https://github.com/GraphNAS/GraphNAS with some changes.
import typing as _typ
import torch
import torch.nn.functional as F
from torch import nn
from .nni import mutables
from .base import BaseSpace
from .operation import act_map, gnn_map
from autogllight.utils.backend import BackendOperator as BK

GRAPHNAS_DEFAULT_GNN_OPS = [
    "gat_8",  # GAT with 8 heads
    "gat_6",  # GAT with 6 heads
    "gat_4",  # GAT with 4 heads
    "gat_2",  # GAT with 2 heads
    "gat_1",  # GAT with 1 heads
    "gcn",  # GCN
    "cheb",  # chebnet
    "sage",  # sage
    "arma",
    "sg",  # simplifying gcn
    "linear",  # skip connection
    "zero",  # skip connection
]

GRAPHNAS_DEFAULT_ACT_OPS = [
    # "sigmoid", "tanh", "relu", "linear",
    #  "softplus", "leaky_relu", "relu6", "elu"
    "sigmoid",
    "tanh",
    "relu",
    "linear",
    "elu",
]

GRAPHNAS_DEFAULT_CON_OPS = ["add", "product", "concat"]
# GRAPHNAS_DEFAULT_CON_OPS=[ "concat"] # for darts


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


class GraphNasNodeClassificationSpace(BaseSpace):
    def __init__(
        self,
        hidden_dim: _typ.Optional[int] = 64,
        layer_number: _typ.Optional[int] = 2,
        dropout: _typ.Optional[float] = 0.9,
        input_dim: _typ.Optional[int] = None,
        output_dim: _typ.Optional[int] = None,
        gnn_ops: _typ.Sequence[_typ.Union[str, _typ.Any]] = GRAPHNAS_DEFAULT_GNN_OPS,
        act_ops: _typ.Sequence[_typ.Union[str, _typ.Any]] = GRAPHNAS_DEFAULT_ACT_OPS,
        con_ops: _typ.Sequence[_typ.Union[str, _typ.Any]] = GRAPHNAS_DEFAULT_CON_OPS,
    ):
        super().__init__()
        self.layer_number = layer_number
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gnn_ops = gnn_ops
        self.act_ops = act_ops
        self.con_ops = con_ops
        self.dropout = dropout

    def build_graph(self):
        self.preproc0 = nn.Linear(self.input_dim, self.hidden_dim)
        self.preproc1 = nn.Linear(self.input_dim, self.hidden_dim)
        node_labels = [mutables.InputChoice.NO_KEY, mutables.InputChoice.NO_KEY]
        for layer in range(2, self.layer_number + 2):
            # input choice
            in_key = f"in_{layer}"
            self.setInputChoice(
                layer, choose_from=node_labels, n_chosen=1, key=in_key,
            )

            # operation choice
            op_key = f"op_{layer}"
            op_candidates = [
                gnn_map(op, self.hidden_dim, self.hidden_dim) for op in self.gnn_ops
            ]
            self.setLayerChoice(layer, op_candidates, key=op_key)

            # input choice candidates
            node_labels.append(op_key)

        # activation choice
        self.setLayerChoice(2 * layer, [act_map_nn(a) for a in self.act_ops], key="act")

        # for DARTS, len(con_ops) can only <=1, for dimension problems
        if len(self.con_ops) > 1:
            self.setLayerChoice(2 * layer + 1, map_nn(self.con_ops), key="concat")

        self.classifier1 = nn.Linear(
            self.hidden_dim * self.layer_number, self.output_dim
        )
        self.classifier2 = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, data):
        # x, edges = data.x, data.edge_index  # x [2708,1433] ,[2, 10556]
        x = BK.feat(data)

        x = F.dropout(x, p=self.dropout, training=self.training)
        pprev_, prev_ = self.preproc0(x), self.preproc1(x)
        prev_nodes_out = [pprev_, prev_]
        for layer in range(2, self.layer_number + 2):
            node_in = getattr(self, f"in_{layer}")(prev_nodes_out)
            op = getattr(self, f"op_{layer}")
            node_out = BK.gconv(op, data, node_in)
            prev_nodes_out.append(node_out)
        act = getattr(self, "act")
        if len(self.con_ops) > 1:
            con = getattr(self, "concat")()
        elif len(self.con_ops) == 1:
            con = self.con_ops[0]
        else:
            con = "concat"

        states = prev_nodes_out
        if con == "concat":
            x = torch.cat(states[2:], dim=1)
        else:
            tmp = states[2]
            for i in range(3, len(states)):
                if con == "add":
                    tmp = torch.add(tmp, states[i])
                elif con == "product":
                    tmp = torch.mul(tmp, states[i])
            x = tmp
        x = act(x)
        if con == "concat":
            x = self.classifier1(x)
        else:
            x = self.classifier2(x)
        return F.log_softmax(x, dim=1)
