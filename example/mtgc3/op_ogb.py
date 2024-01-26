import torch
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.nn.conv import GINConv, GCNConv, GATConv, SAGEConv, GraphConv, ChebConv, ARMAConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from torch_scatter import scatter_mean

from ogb.graphproppred.mol_encoder import AtomEncoder,BondEncoder
from torch_geometric.utils import degree

from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)
from torch import Tensor
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

from torch_sparse import SparseTensor, set_diag

class MLPConvMol(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index, edge_attr, bond_encoder):
        return self.linear(x)

    def main_para(self):
        return [self.linear.weight]

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

class GraphConvMol(GraphConv):
    def __init__(self, in_dim, out_dim):
        super().__init__(in_dim, out_dim)
        #self.bond_encoder = BondEncoder(emb_dim = out_dim)
        self.fuse = False

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_attr, 
                edge_weight: OptTensor = None, size: Size = None, bond_encoder = None) -> Tensor:
        """"""
        edge_embedding = bond_encoder(edge_attr)
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=size, edge_attr=edge_embedding)
        out = self.lin_l(out)

        x_r = x[1]
        if x_r is not None:
            out += self.lin_r(x_r)

        return out

    def main_para(self):
        return [self.lin_l.weight, self.lin_r.weight]

    def message(self, x_j: Tensor, edge_weight: OptTensor, edge_attr) -> Tensor:
        return F.relu(x_j + edge_attr) if edge_weight is None else edge_weight.view(-1, 1) * F.relu(x_j + edge_attr)

class ARMAConvMol(ARMAConv):
    def __init__(self, in_dim, out_dim):
        super().__init__(in_dim, out_dim)
        #self.bond_encoder = BondEncoder(emb_dim = out_dim)
        self.fuse = False

    def forward(self, x: Tensor, edge_index: Adj, edge_attr,
                edge_weight=None, bond_encoder = None) -> Tensor:
        edge_embedding = bond_encoder(edge_attr)

        if isinstance(edge_index, Tensor):
            edge_index, edge_weight = gcn_norm(  # yapf: disable
                edge_index, edge_weight, x.size(self.node_dim),
                add_self_loops=False, dtype=x.dtype)

        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(  # yapf: disable
                edge_index, edge_weight, x.size(self.node_dim),
                add_self_loops=False, dtype=x.dtype)

        x = x.unsqueeze(-3)
        out = x
        for t in range(self.num_layers):
            if t == 0:
                out = out @ self.init_weight
            else:
                out = out @ self.weight[0 if self.shared_weights else t - 1]

            # propagate_type: (x: Tensor, edge_weight: OptTensor)
            out = self.propagate(edge_index, x=out, edge_weight=edge_weight, edge_attr=edge_embedding,
                                 size=None)

            root = F.dropout(x, p=self.dropout, training=self.training)
            out += root @ self.root_weight[0 if self.shared_weights else t]

            if self.bias is not None:
                out += self.bias[0 if self.shared_weights else t]

            if self.act is not None:
                out = self.act(out)

        return out.mean(dim=-3)

    def main_para(self):
        return [self.init_weight, self.root_weight]

    def message(self, x_j: Tensor, edge_weight: Tensor, edge_attr) -> Tensor:
        return edge_weight.view(-1, 1) * (x_j + edge_attr)

class ChebConvMol(ChebConv):
    def __init__(self, in_dim, out_dim):
        super().__init__(in_dim, out_dim, K=2)
        #self.bond_encoder = BondEncoder(emb_dim = out_dim)
        self.fuse = False

    def forward(self, x, edge_index, edge_attr, edge_weight: OptTensor = None,
                batch: OptTensor = None, lambda_max: OptTensor = None, bond_encoder = None):
        edge_embedding = bond_encoder(edge_attr)

        if self.normalization != 'sym' and lambda_max is None:
            raise ValueError('You need to pass `lambda_max` to `forward() in`'
                             'case the normalization is non-symmetric.')

        if lambda_max is None:
            lambda_max = torch.tensor(2.0, dtype=x.dtype, device=x.device)
        if not isinstance(lambda_max, torch.Tensor):
            lambda_max = torch.tensor(lambda_max, dtype=x.dtype,
                                      device=x.device)
        assert lambda_max is not None

        edge_index, norm = self.__norm__(edge_index, x.size(self.node_dim),
                                         edge_weight, self.normalization,
                                         lambda_max, dtype=x.dtype,
                                         batch=batch)

        Tx_0 = x
        Tx_1 = x  # Dummy.
        out = torch.matmul(Tx_0, self.weight[0])

        # propagate_type: (x: Tensor, norm: Tensor)
        if self.weight.size(0) > 1:
            Tx_1 = self.propagate(edge_index, x=x, norm=norm, edge_attr = edge_embedding, size=None)
            out = out + torch.matmul(Tx_1, self.weight[1])

        for k in range(2, self.weight.size(0)):
            Tx_2 = self.propagate(edge_index, x=Tx_1, norm=norm, edge_attr = edge_embedding, size=None)
            Tx_2 = 2. * Tx_2 - Tx_0
            out = out + torch.matmul(Tx_2, self.weight[k])
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out += self.bias

        return out

    def main_para(self):
        return [self.weight]

    def message(self, x_j, norm, edge_attr):
        print(x_j.shape)
        print(norm.shape)
        print(edge_attr.shape)
        return norm.view(-1, 1) * (x_j + edge_attr)

class SAGEConvMol(SAGEConv):
    def __init__(self, in_dim, out_dim):
        super().__init__(in_dim, out_dim)
        #self.bond_encoder = BondEncoder(emb_dim = out_dim)
        self.fuse = False

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_attr,
                size: Size = None, bond_encoder = None) -> Tensor:
        edge_embedding = bond_encoder(edge_attr)
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size, edge_attr=edge_embedding)
        out = self.lin_l(out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out += self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def main_para(self):
        return [self.lin_l.weight, self.lin_r.weight]

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

class GATConvMol(GATConv):
    def __init__(self, in_dim, out_dim):
        super().__init__(in_dim,out_dim, add_self_loops = False)
        #self.bond_encoder = BondEncoder(emb_dim = out_dim)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_attr, 
                size: Size = None, return_attention_weights=None, bond_encoder = None):
        # type: (Union[Tensor, OptPairTensor], Tensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], Tensor, Size, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, Size, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        edge_embedding = bond_encoder(edge_attr)
        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        alpha_l: OptTensor = None
        alpha_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = x_r = self.lin_l(x).view(-1, H, C)
            alpha_l = (x_l * self.att_l).sum(dim=-1)
            alpha_r = (x_r * self.att_r).sum(dim=-1)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = self.lin_l(x_l).view(-1, H, C)
            alpha_l = (x_l * self.att_l).sum(dim=-1)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)
                alpha_r = (x_r * self.att_r).sum(dim=-1)

        assert x_l is not None
        assert alpha_l is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                if size is not None:
                    num_nodes = min(size[0], size[1])
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        out = self.propagate(edge_index, x=(x_l, x_r),
                             alpha=(alpha_l, alpha_r), size=size, edge_attr = edge_embedding)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def main_para(self):
        return [self.lin_l.weight]

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int], edge_attr) -> Tensor:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return F.relu(x_j + edge_attr.unsqueeze(1)) * alpha.unsqueeze(-1)

### GIN convolution along the graph structure
class GINConvMol(MessagePassing):
    def __init__(self, in_dim, out_dim):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super().__init__(aggr = "add")

        #self.mlp = torch.nn.Linear(emb_dim, emb_dim)
        self.mlp1 = torch.nn.Linear(in_dim, out_dim)
        self.bn = torch.nn.BatchNorm1d(out_dim)
        self.relu = torch.nn.ReLU()
        self.mlp2 = torch.nn.Linear(out_dim, out_dim)
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        #self.bond_encoder = BondEncoder(emb_dim = out_dim)

    def forward(self, x, edge_index, edge_attr, bond_encoder):
        edge_embedding = bond_encoder(edge_attr)
        #print(edge_embedding.shape)
        #print(x.shape)
        out = self.mlp1((1 + self.eps) *x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))
        out = self.bn(out)
        out = self.relu(out)
        out = self.mlp2(out)

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def main_para(self):
        return [self.mlp1.weight, self.mlp2.weight]

    def update(self, aggr_out):
        return aggr_out

### GCN convolution along the graph structure
class GCNConvMol(MessagePassing):
    def __init__(self, in_dim, out_dim):
        super().__init__(aggr='add')

        self.linear = torch.nn.Linear(in_dim, out_dim)
        self.root_emb = torch.nn.Embedding(1, out_dim)
        #self.bond_encoder = BondEncoder(emb_dim = out_dim)

    def forward(self, x, edge_index, edge_attr, bond_encoder):
        x = self.linear(x)
        edge_embedding = bond_encoder(edge_attr)

        row, col = edge_index

        #edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype = x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr = edge_embedding, norm=norm) + F.relu(x + self.root_emb.weight) * 1./deg.view(-1,1)

    def message(self, x_j, edge_attr, norm):
        #print(x_j.shape)
        #print(edge_attr.shape)
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def main_para(self):
        return [self.linear.weight]

    def update(self, aggr_out):
        return aggr_out
