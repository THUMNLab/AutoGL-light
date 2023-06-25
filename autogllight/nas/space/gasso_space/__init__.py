from .message_passing import MessagePassing
from .gcn_conv import GCNConv
from .cheb_conv import ChebConv
from .sage_conv import SAGEConv
from .gat_conv import GATConv
from .gin_conv import GINConv, GINEConv
from .arma_conv import ARMAConv
from .edge_conv import EdgeConv, DynamicEdgeConv
from .ops import ZeroConv, LinearConv, SkipConv
import torch


def gnn_map(gnn_name, in_dim, out_dim, concat=False, bias=True):
    """

    :param gnn_name:
    :param in_dim:
    :param out_dim:
    :param concat: for gat, concat multi-head output or not
    :return: GNN model
    """
    norm = True
    if gnn_name == "gat":
        return GATConv(in_dim, out_dim, 1, bias=bias, concat=False, add_self_loops=norm)
    elif gnn_name == "gcn":
        return GCNConv(in_dim, out_dim, add_self_loops=True, normalize=norm)
    elif gnn_name == "gin":
        return GINConv(torch.nn.Linear(in_dim, out_dim))
    elif gnn_name == "cheb":
        return ChebConv(in_dim, out_dim, K=2, bias=bias)
    elif gnn_name == "sage":
        return SAGEConv(in_dim, out_dim, bias=bias)
    elif gnn_name == "gated":
        return GatedGraphConv(in_dim, out_dim, bias=bias)
    elif gnn_name == "arma":
        return ARMAConv(in_dim, out_dim, bias=bias, normalize=norm)
    elif gnn_name == "sg":
        return SGConv(in_dim, out_dim, bias=bias, normalize=norm)
    elif gnn_name == "linear":
        return LinearConv(in_dim, out_dim, bias=bias)
    elif gnn_name == "skip":
        return SkipConv(in_dim, out_dim, bias=bias)
    elif gnn_name == "zero":
        return ZeroConv(in_dim, out_dim, bias=bias)
    else:
        raise ValueError("No such GNN name")


__all__ = [
    "MessagePassing",
    "GCNConv",
    "ChebConv",
    "SAGEConv",
    "GATConv",
    "GINConv",
    "GINEConv",
    "ARMAConv",
    "EdgeConv",
    "DynamicEdgeConv",
    "gnn_map",
]

classes = __all__
