import typing as _typ
from . import BaseSpace
import torch
from .gasso_space import *
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F

GNN_LIST = [
    "gat",  # GAT with 2 heads
    "gcn",  # GCN
    "gin",  # GIN
    # "cheb",  # chebnet
    "sage",  # sage
    # "arma",
    # "sg",  # simplifying gcn
    "linear",  # skip connection
    # "skip",  # skip connection
    # "zero",  # skip connection
]


class MixedOp(nn.Module):
    def __init__(self, in_c, out_c, gnn_list=GNN_LIST):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        self.gnn_list = gnn_list
        for action in gnn_list:
            self._ops.append(gnn_map(action, in_c, out_c))

    def forward(self, x, edge_index, edge_weight, weights, selected_idx=None):
        gnn_list = self.gnn_list
        if selected_idx is None:
            fin = []
            for w, op, op_name in zip(weights, self._ops, gnn_list):
                """if op_name == "gcn":
                    w = 1.0
                else:
                    continue"""
                if edge_weight == None:
                    fin.append(w * op(x, edge_index))
                else:
                    fin.append(w * op(x, edge_index, edge_weight=edge_weight))
            return sum(fin)
            # return sum(w * op(x, edge_index) for w, op in zip(weights, self._ops))
        else:  # unchosen operations are pruned
            return self._ops[selected_idx](x, edge_index)


def Get_edges(adjs,):
    edges = []
    edges_weights = []
    for adj in adjs:
        edges.append(adj[0])
        edges_weights.append(torch.sigmoid(adj[1]))
    return edges, edges_weights


class CellWS(nn.Module):
    def __init__(self, steps, his_dim, hidden_dim, out_dim, dp, bias=True):
        super(CellWS, self).__init__()
        self.steps = steps
        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        self.use2 = False
        self.dp = 0.8
        for i in range(self.steps):
            if i == 0:
                inpdim = his_dim
            else:
                inpdim = hidden_dim
            if i == self.steps - 1:
                oupdim = out_dim
            else:
                oupdim = hidden_dim
            op = MixedOp(inpdim, oupdim)
            self._ops.append(op)
            self._bns.append(nn.BatchNorm1d(oupdim))

    def forward(self, x, adjs, weights):
        edges, ews = Get_edges(adjs)
        for i in range(self.steps):
            if i > 0:
                x = F.relu(x)
                x = F.dropout(x, p=self.dp, training=self.training)
            x = self._ops[i](x, edges[i], ews[i], weights[i])  # call the gcn module
        return x


class GassoSpace(BaseSpace):
    def __init__(
        self,
        hidden_dim: _typ.Optional[int] = 64,
        layer_number: _typ.Optional[int] = 2,
        dropout: _typ.Optional[float] = 0.8,
        input_dim: _typ.Optional[int] = None,
        output_dim: _typ.Optional[int] = None,
        ops: _typ.Tuple = GNN_LIST,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.steps = layer_number
        self.dropout = dropout
        self.ops = ops
        self.use_forward = True
        self.dead_tensor = torch.nn.Parameter(
            torch.FloatTensor([1]), requires_grad=True
        )

    def build_graph(self):
        his_dim, cur_dim, hidden_dim, out_dim = (
            self.input_dim,
            self.input_dim,
            self.hidden_dim,
            self.hidden_dim,
        )
        self.cells = nn.ModuleList()
        self.cell = CellWS(
            self.steps, his_dim, hidden_dim, self.output_dim, self.dropout
        )
        his_dim = cur_dim
        cur_dim = self.steps * out_dim
        self.classifier = nn.Linear(cur_dim, self.output_dim)
        self.initialize_alphas()

    # def forward(self, x, adjs):
    def forward(self, data, adjs):
        if self.use_forward:
            # x, adjs = data.x, data.adj
            x = data.x
            x = F.dropout(x, p=self.dropout, training=self.training)

            weights = []
            for j in range(self.steps):
                weights.append(F.softmax(self.alphas_normal[j], dim=-1))

            x = self.cell(x, adjs, weights)
            x = F.log_softmax(x, dim=1)
            self.current_pred = x.detach()
            return x
        else:
            # for i in self.parameters():
            #    print(i)
            x = self.prediction + self.dead_tensor * 0
            return x

    def keep_prediction(self):
        self.prediction = self.current_pred

    """def to(self, *args, **kwargs):
        fin = super().to(*args, **kwargs)
        device = next(fin.parameters()).device
        fin.alphas_normal = [i.to(device) for i in self.alphas_normal]
        return fin"""

    def initialize_alphas(self):
        num_ops = len(self.ops)

        self.alphas_normal = []
        for i in range(self.steps):
            self.alphas_normal.append(
                Variable(1e-3 * torch.randn(num_ops), requires_grad=True)
            )

        self._arch_parameters = [self.alphas_normal]

    def arch_parameters(self):
        return self.alphas_normal

    def parse_model(self, selection):
        self.use_forward = False
        return self.wrap()
