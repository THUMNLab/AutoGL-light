import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, Conv1d, ELU, PReLU

from torch_geometric.nn import SAGEConv, GATConv, JumpingKnowledge
from torch_geometric.nn import GCNConv, GINConv,GraphConv, ChebConv, LEConv,SGConv,DenseSAGEConv,DenseGCNConv,DenseGINConv,DenseGraphConv, ARMAConv, LEConv
from torch_geometric.nn import global_add_pool,global_mean_pool,global_max_pool,global_sort_pool,GlobalAttention,Set2Set
from torch_geometric.nn import SAGPooling,TopKPooling,EdgePooling,ASAPooling,dense_diff_pool
from torch_geometric.nn.inits import reset
from op_ogb import *
NA_OPS = {
    #SANE
  'sage': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'sage'),
  'sage_sum': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'sum'),
  'sage_max': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'max'),
  'gcn': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'gcn'),
  'gat': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'gat'),
  'gin': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'gin'),
  'cheb': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'cheb'),
  'arma': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'arma'),
  'gat_sym': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'gat_sym'),
  'gat_linear': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'linear'),
  'gat_cos': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'cos'),
  'gat_generalized_linear': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'generalized_linear'),
  #'geniepath': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'geniepath'),
  'mlp': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'mlp'),

  'gcnmol': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'gcnmol'),
  'gatmol': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'gatmol'),
  'ginmol': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'ginmol'),
  'sagemol': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'sagemol'),
  'graphmol': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'graphmol'),
  'chebmol': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'chebmol'),
  'armamol': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'armamol'),
  'mlpmol': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'mlpmol'),

  #graph classification:
  'graphconv_add': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'graphconv_add'),
  'graphconv_mean': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'graphconv_mean'),
  'graphconv_max': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'graphconv_max'),
  #'sgc': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'sgc'),
  'leconv': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'leconv'),

}
POOL_OPS = {

  'hoppool_1': lambda hidden,ratio,num_nodes:Pooling_func(hidden,ratio,'hoppool_1',num_nodes=num_nodes),
  'hoppool_2': lambda hidden,ratio,num_nodes:Pooling_func(hidden,ratio,'hoppool_2',num_nodes=num_nodes),
  'hoppool_3': lambda hidden,ratio,num_nodes:Pooling_func(hidden,ratio,'hoppool_3',num_nodes=num_nodes),

  'mlppool': lambda hidden, ratio, num_nodes: Pooling_func(hidden, ratio, 'mlppool', num_nodes=num_nodes),
  'topkpool': lambda hidden, ratio, num_nodes: Pooling_func(hidden, ratio, 'topkpool', num_nodes=num_nodes),

  'gappool': lambda hidden, ratio, num_nodes: Pooling_func(hidden, ratio, 'gappool', num_nodes=num_nodes),

  'asappool': lambda hidden, ratio, num_nodes: Pooling_func(hidden, ratio, 'asappool', num_nodes=num_nodes),
  'sagpool': lambda hidden, ratio, num_nodes: Pooling_func(hidden, ratio, 'sagpool', num_nodes=num_nodes),
  'sag_graphconv': lambda hidden, ratio, num_nodes: Pooling_func(hidden, ratio, 'graphconv', num_nodes=num_nodes),

  'none': lambda hidden,ratio,num_nodes:Pooling_func(hidden,ratio, 'none', num_nodes=num_nodes),
}
READOUT_OPS = {
    "global_mean": lambda hidden :Readout_func('mean', hidden),
    "global_sum": lambda hidden  :Readout_func('add', hidden),
    "global_max": lambda hidden  :Readout_func('max', hidden),
    "none":lambda hidden  :Readout_func('none', hidden),
    'global_att': lambda hidden  :Readout_func('att', hidden),
    'global_sort': lambda hidden  :Readout_func('sort',hidden),
    'set2set': lambda hidden  :Readout_func('set2set',hidden),
}


LA_OPS={
  'l_max': lambda hidden_size, num_layers: LaAggregator('max', hidden_size, num_layers),
  'l_concat': lambda hidden_size, num_layers: LaAggregator('cat', hidden_size, num_layers),
  'l_mean': lambda hidden_size, num_layers: LaAggregator('mean', hidden_size, num_layers),
  'l_sum': lambda hidden_size, num_layers: LaAggregator('sum', hidden_size, num_layers),
  'l_lstm': lambda hidden_size, num_layers: LaAggregator('lstm', hidden_size, num_layers)
  #min/max
}

class NaAggregator(nn.Module):

  def __init__(self, in_dim, out_dim, aggregator):
    super(NaAggregator, self).__init__()
    #aggregator, K = agg_str.split('_')
    self.aggregator = aggregator
    if aggregator =='mlp':
      self._op = Sequential(Linear(in_dim, out_dim), ELU(), Linear(out_dim, out_dim))
    elif aggregator =='gcnmol':
        self._op = GCNConvMol(in_dim, out_dim)
    elif aggregator =='gatmol':
        self._op = GATConvMol(in_dim, out_dim)
    elif aggregator =='ginmol':
        self._op = GINConvMol(in_dim, out_dim)
    elif aggregator =='sagemol':
        self._op = SAGEConvMol(in_dim, out_dim)
    elif aggregator =='graphmol':
        self._op = GraphConvMol(in_dim, out_dim)
    elif aggregator =='chebmol':
        self._op = ChebConvMol(in_dim, out_dim)
    elif aggregator =='armamol':
        self._op = ARMAConvMol(in_dim, out_dim)
    elif aggregator =='mlpmol':
        self._op = MLPConvMol(in_dim, out_dim)

    self.mol = 'mol' in aggregator

  def reset_params(self):
    if self.aggregator == 'mlp':
      reset(self._op)
    else:
      self._op.reset_parameters()

  def forward(self, x, edge_index, edge_weight=None, edge_attr=None, bond_encoder = None):
    if self.aggregator == 'mlp':
      return self._op(x)
    elif self.mol:
      return self._op(x, edge_index, edge_attr=edge_attr, bond_encoder = bond_encoder)
    else:
      return self._op(x, edge_index, edge_weight=edge_weight)

class Readout_trivial(nn.Module):
  def forward(self, x, batch):
      a = global_mean_pool(x, batch)
      b = global_max_pool(x, batch)
      return torch.cat((a,b), 1)

class Pooling_trivial(nn.Module):
    def __init__(self, in_channels, ratio=0.5, gnn_type='gcn'):
        super().__init__()
        self.op = SAGPooling(in_channels, ratio=ratio, GNN=GCNConv)

    def forward(self, x, edge_index, edge_weight=None, edge_attr=None, batch=None, attn=None, add_self_loop=False, remove_self_loop=False, ft=False):
        x, edge_index, edge_attr, batch, perm, score = self.op(x, edge_index, None, batch, attn)
        #print(x.size(), batch.size())
        return x, edge_index, edge_attr, batch, perm
