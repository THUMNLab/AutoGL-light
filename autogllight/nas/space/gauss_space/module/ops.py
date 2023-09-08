import torch
from torch import nn
import torch_geometric.nn as pygnn
import random

import dgl.function as fn
from dgl.ops import edge_softmax
from dgl.utils import expand_as_pair
import dgl.nn.pytorch as dglnn

class LinearConv(torch.nn.Linear):
    def forward(self, x, *args, **kwargs):
        if isinstance(x, tuple): x = x[1]
        return super().forward(x)

class IdentityConv(torch.nn.Identity):
    def forward(self, x, *args, **kwargs):
        if isinstance(x, tuple): x = x[1]
        return super().forward(x)
    
    def reset_parameters(self):
        return

class ZeroConv(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.tensor([0.]), True)
    
    def forward(self, x, *args, **kwargs):
        if isinstance(x, tuple): x = x[1]
        return torch.zeros_like(x).to(x.device) + 0.0 * self.dummy
    
    def reset_parameters(self):
        return

class Wrap(torch.nn.Module):
    def __init__(self, indim, outdim, module) -> None:
        super().__init__()
        if indim != outdim:
            self.map = torch.nn.Linear(indim, outdim)
        else:
            self.map = torch.nn.Identity()
        self.core = module
    
    def forward(self, x, edge_index):
        x = self.map(x)
        return self.core(x, edge_index)
    
    def reset_parameters(self):
        if hasattr(self.map, 'reset_parameters'):
            self.map.reset_parameters()
        self.core.reset_parameters()

class MLP(torch.nn.Module):
    r"""A multi-layer perception (MLP) model.

    Args:
        channel_list (List[int]): List of input, intermediate and output
            channels. :obj:`len(channel_list) - 1` denotes the number of layers
            of the MLP.
        dropout (float, optional): Dropout probability of each hidden
            embedding. (default: :obj:`0.`)
        batch_norm (bool, optional): If set to :obj:`False`, will not make use
            of batch normalization. (default: :obj:`True`)
        relu_first (bool, optional): If set to :obj:`True`, ReLU activation is
            applied before batch normalization. (default: :obj:`False`)
    """
    def __init__(self, channel_list, dropout: float = 0.,
                 batch_norm: bool = True, relu_first: bool = False):
        super().__init__()
        assert len(channel_list) >= 2
        self.channel_list = channel_list
        self.dropout = dropout
        self.relu_first = relu_first

        self.lins = torch.nn.ModuleList()
        for dims in zip(channel_list[:-1], channel_list[1:]):
            self.lins.append(torch.nn.Linear(*dims))

        self.norms = torch.nn.ModuleList()
        for dim in zip(channel_list[1:-1]):
            self.norms.append(torch.nn.BatchNorm1d(dim) if batch_norm else torch.nn.Identity())

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for norm in self.norms:
            if hasattr(norm, 'reset_parameters'):
                norm.reset_parameters()


    def forward(self, x):
        """"""
        x = self.lins[0](x)
        for lin, norm in zip(self.lins[1:], self.norms):
            if self.relu_first:
                x = x.relu_()
            x = norm(x)
            if not self.relu_first:
                x = x.relu_()
            x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
            x = lin.forward(x)
        return x


    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({str(self.channel_list)[1:-1]})'

# define operators
def op(name, num_input, num_output=None, dropout=0.5, **kwargs):
    num_output = num_output or num_input
    if name == "gcn": # currently no support for neighbor sampling
        return pygnn.GCNConv(num_input, num_output)
    if name == "gat1":
        return pygnn.GATConv(num_input, num_output // 1, heads=1)
    if name == "gat2":
        return pygnn.GATConv(num_input, num_output // 2, heads=2)
    if name == "gat4":
        return pygnn.GATConv(num_input, num_output // 4, heads=4)
    if name == "gat8":
        return pygnn.GATConv(num_input, num_output // 8, heads=8)
    if name == "gat16":
        return pygnn.GATConv(num_input, num_output // 16, heads=16)
    if name == "sage":
        return pygnn.SAGEConv(num_input, num_output)
    if name == "cheb": # currently no support for neighbor sampling
        return pygnn.ChebConv(num_input, num_output, 2)
    if name == "graph":
        return pygnn.GraphConv(num_input, num_output, aggr='mean')
    if name == "transformer":
        return pygnn.TransformerConv(num_input, num_output)
    if name == "gin":
        return pygnn.GINConv(nn.Sequential(
            nn.Linear(num_input, num_output),
            nn.ReLU(),
            nn.Linear(num_output, num_output)
        ), aggr='mean')
    if name == 'ginv2':
        return pygnn.GINConv(
            nn.Sequential(
                nn.BatchNorm1d(num_input),
                nn.Linear(num_input, num_output),
                nn.ReLU(),
                nn.Linear(num_output, num_output)
            )
        )
    if name == 'ginv3':
        return pygnn.GINConv(MLP([num_input, num_output, num_output]))
    if name == "resggraph":
        return pygnn.ResGatedGraphConv(num_input, num_output)
    if name == "gatv2":
        return pygnn.GATv2Conv(num_input, num_output)
    if name == "cg":
        return pygnn.CGConv(num_input, num_output)
    if name == "feast":
        return pygnn.FeaStConv(num_input, num_output)
    if name == "le":
        return pygnn.LEConv(num_input, num_output)
    if name == "film":
        return pygnn.FiLMConv(num_input, num_output)
    if name == "linear":
        return LinearConv(num_input, num_output)
    if name == "identity":
        return IdentityConv()
    if name == "zero":
        return ZeroConv()
    if name == 'sgc':
        return pygnn.SGConv(num_input, num_output)
    if name == 'appnp':
        return Wrap(num_input, num_output, pygnn.APPNP(10, 0.2))

def uniform_sample(layers, space):
    if isinstance(space[0], list):
        assert len(space) == layers
        return [random.choice(s) for s in space]
    return [random.choice(space) for _ in range(layers)]

def uniform_samples(layers, space, number):
    archs = []
    while len(archs) < number:
        arch = uniform_sample(layers, space)
        if arch not in archs: archs.append(arch)
    return archs

def mutate(arch, space, ratio="auto"):
    if ratio == "auto":
        ratio = 1 - (0.5 ** (1 / len(arch)))
    new_arch = []
    for i, a in enumerate(arch):
        if random.random() < ratio:
            # mutate
            if isinstance(space[0], list):
                new_arch.append(random.choice(space[i]))
            else:
                new_arch.append(random.choice(space))
        else:
            new_arch.append(a)
    if new_arch == arch:
        return mutate(arch, space, ratio)
    return new_arch

# borrowed from baseline
class GATConvDGL(nn.Module):
    def __init__(
        self,
        node_feats,
        edge_feats,
        out_feats,
        n_heads=1,
        attn_drop=0.0,
        edge_drop=0.0,
        negative_slope=0.2,
        residual=True,
        activation=None,
        use_attn_dst=True,
        allow_zero_in_degree=True,
        use_symmetric_norm=False,
    ):
        super(GATConvDGL, self).__init__()
        self._n_heads = n_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(node_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self._use_symmetric_norm = use_symmetric_norm

        # feat fc
        self.src_fc = nn.Linear(self._in_src_feats, out_feats * n_heads, bias=False)
        if residual:
            self.dst_fc = nn.Linear(self._in_src_feats, out_feats * n_heads)
            self.bias = None
        else:
            self.dst_fc = None
            self.bias = nn.Parameter(out_feats * n_heads)

        # attn fc
        self.attn_src_fc = nn.Linear(self._in_src_feats, n_heads, bias=False)
        if use_attn_dst:
            self.attn_dst_fc = nn.Linear(self._in_src_feats, n_heads, bias=False)
        else:
            self.attn_dst_fc = None
        if edge_feats > 0:
            self.attn_edge_fc = nn.Linear(edge_feats, n_heads, bias=False)
        else:
            self.attn_edge_fc = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.edge_drop = edge_drop
        self.leaky_relu = nn.LeakyReLU(negative_slope, inplace=True)
        self.activation = activation

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.src_fc.weight, gain=gain)
        if self.dst_fc is not None:
            nn.init.xavier_normal_(self.dst_fc.weight, gain=gain)

        nn.init.xavier_normal_(self.attn_src_fc.weight, gain=gain)
        if self.attn_dst_fc is not None:
            nn.init.xavier_normal_(self.attn_dst_fc.weight, gain=gain)
        if self.attn_edge_fc is not None:
            nn.init.xavier_normal_(self.attn_edge_fc.weight, gain=gain)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat_src, feat_edge=None):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    assert False

            if graph.is_block:
                feat_dst = feat_src[: graph.number_of_dst_nodes()]
            else:
                feat_dst = feat_src

            if self._use_symmetric_norm:
                degs = graph.srcdata["deg"]
                # degs = graph.out_degrees().float().clamp(min=1)
                norm = torch.pow(degs, -0.5)
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = torch.reshape(norm, shp)
                feat_src = feat_src * norm

            feat_src_fc = self.src_fc(feat_src).view(-1, self._n_heads, self._out_feats)
            feat_dst_fc = self.dst_fc(feat_dst).view(-1, self._n_heads, self._out_feats)
            attn_src = self.attn_src_fc(feat_src).view(-1, self._n_heads, 1)

            # NOTE: GAT paper uses "first concatenation then linear projection"
            # to compute attention scores, while ours is "first projection then
            # addition", the two approaches are mathematically equivalent:
            # We decompose the weight vector a mentioned in the paper into
            # [a_l || a_r], then
            # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
            # Our implementation is much efficient because we do not need to
            # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
            # addition could be optimized with DGL's built-in function u_add_v,
            # which further speeds up computation and saves memory footprint.
            graph.srcdata.update({"feat_src_fc": feat_src_fc, "attn_src": attn_src})

            if self.attn_dst_fc is not None:
                attn_dst = self.attn_dst_fc(feat_dst).view(-1, self._n_heads, 1)
                graph.dstdata.update({"attn_dst": attn_dst})
                graph.apply_edges(fn.u_add_v("attn_src", "attn_dst", "attn_node"))
            else:
                graph.apply_edges(fn.copy_u("attn_src", "attn_node"))

            e = graph.edata["attn_node"]
            if feat_edge is not None:
                attn_edge = self.attn_edge_fc(feat_edge).view(-1, self._n_heads, 1)
                graph.edata.update({"attn_edge": attn_edge})
                e += graph.edata["attn_edge"]
            e = self.leaky_relu(e)

            if self.training and self.edge_drop > 0:
                perm = torch.randperm(graph.number_of_edges(), device=e.device)
                bound = int(graph.number_of_edges() * self.edge_drop)
                eids = perm[bound:]
                graph.edata["a"] = torch.zeros_like(e)
                graph.edata["a"][eids] = self.attn_drop(edge_softmax(graph, e[eids], eids=eids))
            else:
                graph.edata["a"] = self.attn_drop(edge_softmax(graph, e))

            # message passing
            graph.update_all(fn.u_mul_e("feat_src_fc", "a", "m"), fn.sum("m", "feat_src_fc"))

            rst = graph.dstdata["feat_src_fc"]

            if self._use_symmetric_norm:
                degs = graph.dstdata["deg"]
                # degs = graph.in_degrees().float().clamp(min=1)
                norm = torch.pow(degs, 0.5)
                shp = norm.shape + (1,) * (feat_dst.dim())
                norm = torch.reshape(norm, shp)
                rst = rst * norm

            # residual
            if self.dst_fc is not None:
                rst += feat_dst_fc
            else:
                rst += self.bias

            # activation
            if self.activation is not None:
                rst = self.activation(rst, inplace=True)

            return rst.flatten(1, -1)

class EdgeConv(nn.Module):
    def __init__(self, conv) -> None:
        super().__init__()
        self.conv = conv
    
    def forward(self, g, node_feat, edge_feat):
        return self.conv(g, node_feat)


class Linear(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super().__init__()
        self.core = nn.Linear(in_dim, out_dim)
    
    def forward(self, g, node_feat, edge_feat):
        return self.core(node_feat)


def op_dgl(name, in_dim, out_dim, edge_feat, **kwargs):
    if name == 'gat':
        return GATConvDGL(
            in_dim,
            edge_feat,
            out_dim // kwargs['n_heads'],
            n_heads=kwargs['n_heads'],
            attn_drop=kwargs['attn_drop'],
            edge_drop=kwargs['edge_drop'],
            use_attn_dst=kwargs['use_attn_dst'],
            allow_zero_in_degree=kwargs['allow_zero_in_degree'],
            use_symmetric_norm=False
        )
    if name == 'gcn':
        return EdgeConv(dglnn.GraphConv(in_dim, out_dim, allow_zero_in_degree=kwargs['allow_zero_in_degree']))
    if name == 'tag':
        return EdgeConv(dglnn.TAGConv(in_dim, out_dim))
    if name == 'edge':
        return EdgeConv(dglnn.EdgeConv(in_dim, out_dim, allow_zero_in_degree=kwargs['allow_zero_in_degree']))
    if name == 'sage':
        return EdgeConv(dglnn.SAGEConv(in_dim, out_dim, 'mean'))
    if name == 'sg':
        return EdgeConv(dglnn.SGConv(in_dim, out_dim, allow_zero_in_degree=kwargs['allow_zero_in_degree']))
    if name == 'appnp':
        return EdgeConv(dglnn.APPNPConv(10, 0.1))
    if name == 'gin':
        return EdgeConv(dglnn.GINConv(nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        ), 'max'))
    if name == 'gmm':
        return dglnn.GMMConv(in_dim, out_dim, edge_feat, 5)
    if name == 'cheb':
        return EdgeConv(dglnn.ChebConv(in_dim, out_dim, 2))
    if name == 'agnn':
        return EdgeConv(dglnn.AGNNConv(allow_zero_in_degree=kwargs['allow_zero_in_degree']))
    if name == 'cf':
        return dglnn.CFConv(in_dim, edge_feat, out_dim, out_dim)
    if name == 'linear':
        return Linear(in_dim, out_dim)
