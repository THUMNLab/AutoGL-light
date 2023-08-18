from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')


NA_PRIMITIVES_o = [
  'sage',
  'gcn',
  'gin',
  'gat',
  'graphconv_mean',  # aggr:add mean max
  'mlp',
]

NA_PRIMITIVES = [
  'gcnmol',
  'ginmol',
  'gatmol',
  'sagemol',
  'graphmol',  # aggr:add mean max
  'mlpmol',
]

POOL_PRIMITIVES=[
  'none',
]

POOL_PRIMITIVES_o=[
  'hoppool_1',
  'hoppool_2',
  'hoppool_3',

  'topkpool',
  'mlppool',

  'gappool',

  'sagpool',
  'asappool',
  'sag_graphconv', #GCPOOL

  'none',
]
READOUT_PRIMITIVES = [
  'global_mean',
  'global_max',
  'global_sum',
  'none',
  # 'mean_max',
  'global_att',
'global_sort',#DGCNN
  'set2set',  # a seq2seq method
]
ACT_PRIMITIVES = [
  "sigmoid", "tanh", "relu",
  "softplus", "leaky_relu", "relu6", "elu"
]
LA_PRIMITIVES=[
  'l_max',
  'l_concat',
  'l_lstm',
  'l_sum',
  'l_mean',
]


