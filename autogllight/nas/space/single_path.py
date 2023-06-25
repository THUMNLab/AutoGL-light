from .operation import gnn_map
import typing as _typ
import torch

import torch.nn.functional as F

from .base import BaseSpace
from ...utils.backend import BackendOperator as BK


class SinglePathNodeClassificationSpace(BaseSpace):
    def __init__(
        self,
        hidden_dim: _typ.Optional[int] = 64,
        layer_number: _typ.Optional[int] = 2,
        dropout: _typ.Optional[float] = 0.2,
        input_dim: _typ.Optional[int] = None,
        output_dim: _typ.Optional[int] = None,
        ops: _typ.Tuple = ["gcn", "gat_8"],
    ):
        super().__init__()
        self.layer_number = layer_number
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.ops = ops
        self.dropout = dropout

    def build_graph(self):
        for layer in range(self.layer_number):
            key = f"op_{layer}"
            in_dim = self.input_dim if layer == 0 else self.hidden_dim
            out_dim = (
                self.output_dim if layer == self.layer_number - 1 else self.hidden_dim
            )
            op_candidates = [
                op(in_dim, out_dim)
                if isinstance(op, type)
                else gnn_map(op, in_dim, out_dim)
                for op in self.ops
            ]
            self.setLayerChoice(layer, op_candidates, key=key)

    def forward(self, data):
        x = BK.feat(data)
        for layer in range(self.layer_number):
            op = getattr(self, f"op_{layer}")
            x = BK.gconv(op, data, x)
            if layer != self.layer_number - 1:
                x = F.leaky_relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)
