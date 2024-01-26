import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.utils import add_self_loops, remove_self_loops
# from operations import *
from op_graph_classification import *
from torch.autograd import Variable
from torch_geometric.nn import  global_mean_pool, global_add_pool
from archgen import AG, Collect_main_para, Apply_mask, StructureMask
#import sys
#sys.path.append("../")
from ogb.graphproppred.mol_encoder import AtomEncoder,BondEncoder
import random

NA_PRIMITIVES= [
  'gcnmol',
  'ginmol',
  'gatmol',
  'sagemol',
  'graphmol',  # aggr:add mean max
  #'chebmol',
  'armamol',
  'mlpmol',
]

class NaMixedOp(nn.Module):

  def __init__(self, in_dim, out_dim, n_chunk, chunk_size, with_linear):
    super(NaMixedOp, self).__init__()
    self._ops = nn.ModuleDict()

    self.num_ops = len(NA_PRIMITIVES)
    self.n_chunk = n_chunk
    self.chunk_size = chunk_size
    self.out_dim = out_dim
    self.set = "train_no"
    self.ln = nn.LayerNorm(out_dim)
    self.chunk_bond_encoder = nn.ModuleDict()

    self.use_global_encoder = ['ginmol', 'sagemol', 'graphmol']
    for primitive in NA_PRIMITIVES:
      if primitive not in self.use_global_encoder:
        self.chunk_bond_encoder[primitive] = BondEncoder(chunk_size)
      op = NA_OPS[primitive](in_dim, out_dim)
      Collect_main_para(op._op, in_dim, out_dim)
      self._ops[primitive] = op

      if with_linear:
        self._ops_linear = nn.ModuleList()
        op_linear = torch.nn.Linear(in_dim, out_dim)
        self._ops_linear.append(op_linear)

      # self.act = act_map(act)

  def dl_df(self):
    g = self.fi.grad # batch * n_c * chunksize
    g = g.norm(2) 
    return g

  def forward(self, x, weights, edge_index, edge_weights, edge_attr, with_linear, chunk_eye=None, graph_match = False, mask = None, bond_encoder = None):
    mixed_res = []
    for op in self._ops:
        Apply_mask(self._ops[op]._op, mask)
    for opname in self._ops:
        op = self._ops[opname]
        if opname in self.use_global_encoder:
            be = bond_encoder[opname]
        else:
            be = self.chunk_bond_encoder[opname]
        mixed_res.append(op(x, edge_index, edge_weight=edge_weights, edge_attr=edge_attr, bond_encoder = be))
        # print('without linear')

    chunk_eye = chunk_eye.to(weights.device)
    if graph_match:
        # weights: op * n_c
        opmap = torch.stack(mixed_res, 1) # batch * op * dim
        opmap = self.ln(opmap)
        opmap = opmap.reshape(-1, self.num_ops, self.chunk_size) # batch * op * n_c * chunk_size
        #weights = weights.reshape(1, self.num_ops, self.n_chunk, 1) # 1 * op * n_c *1
        weights = weights.unsqueeze(0).unsqueeze(-1) # 1 * op * n_c * 1
        res = opmap * weights # batch * op * n_c * chunk_size
        res = res.sum(1) # batch * n_c * chunk_size
        if self.training:
            self.fi = res
            self.fi.retain_grad()
        res = res.reshape(-1, self.out_dim) # batch * dim

        #diagl = self.diagonal_loss(chunk_eye)
        return res

        # Below: GRACES
        """
        # weights: batch * op
        opmap = torch.stack(mixed_res, 1) # batch * op * dim
        weights = weights.unsqueeze(1) # batch * 1 * op
        res = weights @ opmap # batch * 1 * dim
        res = res.squeeze() # batch * dim
        return res
        """

    else:
        return sum([w * op for w, op in zip(weights, mixed_res)])

class MixHead(nn.Linear):
    def __init__(self, in_dim, out_dim, ratio_1):
        super().__init__(in_dim, out_dim)
        self.mask = torch.nn.Parameter(torch.bernoulli(torch.ones_like(self.weight) * ratio_1), requires_grad=False)
        return
        self.mask = torch.zeros_like(self.weight)
        for i in range(0, out_dim):
            a1 = random.randint(0, in_dim - 1)
            a2 = random.randint(0, in_dim - 1)
            self.mask[i, a1] = 1
            self.mask[i, a2] = 1
        self.mask = torch.nn.Parameter(self.mask, requires_grad=False)

    def forward(self, input):
        weight = self.weight * self.mask
        return F.linear(input, weight, self.bias)

class MixHeadAblation(nn.Linear):
    def __init__(self, in_dim, out_dim, num_layers, n_chunks, chunk_size):
        super().__init__(in_dim, out_dim)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.n_chunks = n_chunks
        self.chunk_size = chunk_size
        self.mask = torch.zeros(self.n_chunks, out_dim)
        for i in range(0, self.out_dim):
            a1 = random.randint(0, self.n_chunks - 1)
            a2 = random.randint(0, self.n_chunks - 1)
            self.mask[a1, i] = 1
            self.mask[a2, i] = 1
        self.mask = torch.nn.Parameter(self.mask, requires_grad=False)

    def forward(self, input):
        weight = self.weight.reshape(2 * (self.num_layers + 1), self.n_chunks, self.chunk_size, self.out_dim)
        mask = self.mask.reshape(1, self.n_chunks, 1, self.out_dim)
        weight = weight * mask
        weight = weight.reshape(self.out_dim, self.in_dim)
        return F.linear(input, weight, self.bias)
        
class Network(nn.Module):

  def __init__(self, criterion, in_dim, out_dim, hidden_size, num_layers, n_chunks, dropout=0.5, epsilon=0.0, args=None, with_conv_linear=False,num_nodes=0, mol=False, virtual=False):
    super(Network, self).__init__()
    self.separate_head = args.sep_head
    self.in_dim = in_dim
    self.out_dim = out_dim
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.num_nodes = num_nodes
    self.n_chunks = n_chunks
    self.chunk_size = hidden_size // n_chunks
    self._criterion = criterion
    self.dropout = dropout
    self.epsilon = epsilon
    self.with_linear = with_conv_linear
    self.explore_num = 0
    self.args = args
    self.temp = args.temp
    self._loc_mean = args.loc_mean
    self._loc_std = args.loc_std
    self.mol = mol # if the task is molecule
    self.virtual = virtual
    self.use_global_encoder = ['ginmol', 'sagemol', 'graphmol']
    self.chunk_eye = torch.eye(self.n_chunks).tile(self.chunk_size, self.chunk_size).reshape(self.chunk_size, self.n_chunks, self.chunk_size, self.n_chunks).permute(1, 0, 3, 2).reshape(self.hidden_size, -1)
    self.chunk_eye = torch.ones_like(self.chunk_eye) - self.chunk_eye
    self.uw = torch.nn.Parameter(torch.zeros(out_dim), requires_grad=True)

    self.lin1 = nn.Linear(in_dim, hidden_size)
    self.atom_encoder = AtomEncoder(hidden_size)
    self.virtualnode_embedding = torch.nn.Embedding(1, hidden_size)
    torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

    self.mlp_virtualnode_list = torch.nn.ModuleList()
    for layer in range(num_layers - 1):
        self.mlp_virtualnode_list.append(torch.nn.Sequential(torch.nn.Linear(hidden_size, 2*hidden_size), torch.nn.BatchNorm1d(2*hidden_size), torch.nn.ReLU(), \
                                                torch.nn.Linear(2*hidden_size, hidden_size), torch.nn.BatchNorm1d(hidden_size), torch.nn.ReLU()))

    # node aggregator op
    self.gnn_layers = nn.ModuleList()
    self.bns = nn.ModuleList()
    self.bond_encoders = nn.ModuleList()
    for i in range(num_layers):
        mli = nn.ModuleList()
        for j in range(n_chunks):
            mli.append(NaMixedOp(hidden_size, self.chunk_size, self.n_chunks, self.chunk_size, self.with_linear))
        self.gnn_layers.append(mli)
        self.bns.append(nn.BatchNorm1d(hidden_size))

        bedi = nn.ModuleDict()
        for op in NA_PRIMITIVES:
            if op in self.use_global_encoder:
                bedi[op] = BondEncoder(hidden_size)
        self.bond_encoders.append(bedi)

    self.stru_masks = []
    for i in range(num_layers):
        sm = StructureMask(args, hidden_size, n_chunks, args.stru_size)
        sm.to(self.args.device)
        self.stru_masks.append(sm)

    #self.pooling_global = PoolingMixedOp(hidden_size * (num_layers + 1), args.pooling_ratio, num_nodes=self.num_nodes)
    self.pooling_trivial = Pooling_trivial(hidden_size * (num_layers + 1), args.pooling_ratio)

    self.layer7 = Readout_trivial()

    # Output for short task: #task * chunk_size
    #self.classifier = nn.Linear(self.hidden_size * 2, self.n_chunks)
    if self.separate_head:
        # separate head
        #self.classifier = nn.Linear(self.chunk_size * 2, self.n_chunks)
        self.classifier = nn.Linear(self.chunk_size * (num_layers + 1) * 2, self.n_chunks)
    else:
        # mix head
        self.classifier = MixHead(self.hidden_size * (num_layers + 1) * 2, out_dim, 1 / n_chunks)
        #self.classifier = MixHeadAblation(self.hidden_size * (num_layers + 1) * 2, out_dim, self.num_layers, self.n_chunks, self.chunk_size)

    # ablation: nohead
    #self.separate_head = False
    #self.classifier = nn.Linear(self.hidden_size * (num_layers + 1) * 2, out_dim)

    self._initialize_alphas()

  def stru_paras(self):
    for i in self.stru_masks:
        for j in i.parameters():
            yield j

  def set_masks(self, mask):
      for module in self.modules():
          if isinstance(module, MessagePassing):
              module.__explain__ = True
              module.__edge_mask__ = mask

  def _get_categ_mask(self, alpha):
      # log_alpha = torch.log(alpha)
      log_alpha = alpha
      u = torch.zeros_like(log_alpha).uniform_()
      softmax = torch.nn.Softmax(-1)
      one_hot = softmax((log_alpha + (-((-(u.log())).log()))) / self.temp)
      return one_hot

  def get_one_hot_alpha(self, alpha):
      one_hot_alpha = torch.zeros_like(alpha, device=alpha.device)
      idx = torch.argmax(alpha, dim=-1)

      for i in range(one_hot_alpha.size(0)):
        one_hot_alpha[i, idx[i]] = 1.0

      return one_hot_alpha

  def get_uniform_alpha(self, alpha):
      uniform_alpha = torch.full_like(alpha, 1.0 / alpha.size(-1), device=alpha.device)
      return uniform_alpha

  def forward(self, data, mode='none', graph_alpha = None, masks = None):
    self.args.search_act = False
    with_linear = self.with_linear
    x, edge_index = data.x, data.edge_index
    edge_attr = getattr(data, 'edge_attr', None)
    batch = data.batch
    # edge_index, _ = remove_self_loops(edge_index)
    if edge_attr == None:
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size()[0])

    if self.args.model_type == 'darts':
        na_alphas = F.softmax(self.log_na_alphas, dim=-1)
        # print('DARTS: sampled arch in train w', self._sparse(na_alphas, act_alphas, pool_alphas, readout_alphas, la_alphas))
    else:
        na_alphas = self._get_categ_mask(self.log_na_alphas)
        # print('alpha in train w:',self._arch_parameters)
        # print('sampled arch in train w', self._sparse(na_alphas, act_alphas, pool_alphas, readout_alphas, la_alphas))

    if mode == 'evaluate_single_path':
        na_alphas = self.get_one_hot_alpha(na_alphas)

    elif mode == 'mixed':
        na_alphas = self.get_uniform_alpha(na_alphas)
        
    elif mode == 'mads':
        # use index_select
        #print(len(graph_alpha))
        #print(graph_alpha[0].shape)
        na_alphas = graph_alpha
        #batch_weight = torch.index_select(weights, 0, batch)

    graph_representations = []
    if not self.mol:
        x = F.elu(self.lin1(x))
    else:
        x = self.atom_encoder(x)

    edge_weights = torch.ones(edge_index.size()[1], device=edge_index.device).float()
    virtualnode_embedding = self.virtualnode_embedding(torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))

    #graph_representations.append(self.readout_layers[0](x, batch, None, readout_alphas[0]))
    gr = [x]
    dls = []

    for i in range(self.num_layers):
        if self.virtual:
            orix = x
            x = x + virtualnode_embedding[batch]
        if mode == "mads":
            stru_weights = self.stru_masks[i](x, edge_index, self.eta)
            bond_encoder = self.bond_encoders[i]
            xs = []
            for j in range(self.n_chunks):
                stru_weight = stru_weights[i]
                #print('edge', edge_index.shape)
                self.set_masks(stru_weight)
                x_chunk = self.gnn_layers[i][j](x, na_alphas[i][:,j], edge_index,
                    edge_weights, edge_attr, with_linear, self.chunk_eye, 
                    graph_match = True, mask = masks[i][: , self.chunk_size * j : self.chunk_size * (j+1)],
                    bond_encoder = bond_encoder)
                #print(x.shape)
                xs.append(x_chunk)
            x = torch.hstack(xs)
            #print(x.shape)
        else:
            x, dl = self.gnn_layers[i](x, na_alphas[i], edge_index, edge_weights, edge_attr, with_linear)

        x = self.bns[i](x)
        #layer_norm = nn.LayerNorm(normalized_shape=x.size(), elementwise_affine=False)
        #x = layer_norm(x)
        if i < self.num_layers - 1:
            x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        gr.append(x)

        #x, edge_index, edge_weights, batch, _ = self.pooling_layers[i](x, edge_index, edge_weights, data, batch, None, pool_alphas[i])
        #graph_representations.append(self.readout_layers[i+1](x, batch, None, readout_alphas[i+1]))

        if self.virtual and i < self.num_layers - 1:
            virtualnode_embedding_temp = global_add_pool(orix, batch) + virtualnode_embedding
            ### transform virtual nodes using MLP
            virtualnode_embedding = F.dropout(self.mlp_virtualnode_list[i](virtualnode_embedding_temp), self.dropout, training = self.training)

    x = torch.cat(gr, 1)
    #x = gr[-1]

    # if use pooling trivial, you should set the GNN in SAG in op_graph_classification.py 
    x, edge_index, edge_weights, batch, _ = self.pooling_trivial(x, edge_index, edge_weights, data, batch, None)
        
    x5 = self.layer7(x, batch)
    #x5 = self.layer6(graph_representations, la_alphas[0])
    #x5 = F.elu(self.lin_output(x5))

    # GRACES
    """x_emb = self.lin_output(x5)
    x5 = F.elu(x_emb)
    x5 = F.dropout(x5, p=self.dropout, training=self.training)
    logits = self.classifier(x5)"""

    # TARA
    if self.separate_head:
        x5 = x5.reshape(-1, 2, self.num_layers + 1, self.n_chunks, self.chunk_size)
        x5 = x5.permute(0, 3, 2, 1, 4)
        x5 = x5.reshape(-1, self.n_chunks, 2 * self.chunk_size * (self.num_layers + 1)) 
        logits = self.classifier(x5) # bs * n_chunk * n_chunk
        logits = logits.diagonal(offset = 0, dim1 = 1, dim2 = 2)
        #logits = self.classifier(x5)
    else:
        logits = self.classifier(x5)

    if not self.mol:
        logits = F.log_softmax(logits, dim = -1)

    return logits, 0

  def _initialize_alphas(self):
    num_na_ops = len(NA_PRIMITIVES)
    """if self.args.model_type == 'mads':
        self.mhssl = MhsslBox(self.args, num_na_ops, num_pool_ops)
        self._arch_parameters = self.mhssl.it.parameters()
        return"""

    if self.args.model_type == 'darts':
        self.log_na_alphas = Variable(1e-3*torch.randn(self.num_layers, num_na_ops).cuda(), requires_grad=True)
    else:
        self.log_na_alphas = Variable(torch.ones(self.num_layers, num_na_ops).normal_(self._loc_mean, self._loc_std).cuda(), requires_grad=True)

    self._arch_parameters = [
      self.log_na_alphas,
    ]

  def arch_parameters(self):
    return self._arch_parameters

  def genotype(self):
    gene = self._sparse(F.softmax(self.log_na_alphas, dim=-1).data.cpu())
    return gene

class MTGC3(nn.Module):
    def __init__(self, criterion, in_dim, out_dim, hidden_size, num_layers=3, n_chunks = 12, dropout=0.5, epsilon=0.0, args=None, with_conv_linear=False, num_nodes=0, mol=False, virtual=False):
        super().__init__()
        num_na_ops = len(NA_PRIMITIVES)
        #num_pool_ops = len(POOL_PRIMITIVES)
        self.args = args
        self.supernet = Network(criterion, in_dim, out_dim, hidden_size, num_layers, n_chunks, dropout, epsilon, args, with_conv_linear, num_nodes, mol, virtual)
        self.ag = AG(args, num_na_ops, n_chunks)
        #self._arch_parameters = self.mhssl.it.parameters()
        self.explore_num = 0

    def forward(self, data):
        # ag
        graph_alpha, masks = self.ag()
        #print(graph_alpha[-1])
        # final supernet
        pred, diagloss = self.supernet(data, mode = 'mads', graph_alpha = graph_alpha, masks = masks)
        return pred

    def currl_forward(self, data, ds, eta):
        #eta = (1 + eta) / 2
        # ag
        graph_alpha, masks = self.ag()

        # currl training
        newmasks = []
        eps = 1e-8
        chunk_size = self.args.hidden_size // self.args.n_chunks
        for mask, dl in zip(masks, ds):
            # mask: n_c * n_c; dl: n_c
            mask = mask.detach()
            dl = dl.detach().to(mask.device)
            dfj = dl.tile((self.args.hidden_size, chunk_size)).reshape(self.args.hidden_size, chunk_size, self.args.n_chunks).permute(0, 2, 1).reshape(self.args.hidden_size, -1)
            dfj = dfj + eps
            #print(dfj.shape)
            #print(dfj)
            dfi = dfj.t()
            newmask = dfj / dfi * torch.tanh(mask * dfi / dfj)
            newmask = (1-eta) * newmask + eta * mask
            newmasks.append(newmask)

        # final supernet
        pred, diagloss = self.supernet(data, mode = 'mads', graph_alpha = graph_alpha, masks = newmasks)
        return pred

    def get_ds(self):
        fin = [torch.FloatTensor([mixop.dl_df() for mixop in i]) for i in self.supernet.gnn_layers]
        # fin [o] * num_layers
        return fin

    def mixed_train(self, data):
        pred = self.supernet(data, mode = 'mixed') # graph * g_d
        return pred