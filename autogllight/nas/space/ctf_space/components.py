import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import JumpingKnowledge
import numpy as np
import random

from .genotypes import *
from .utils import *
from .pyg_gnn_layer import GeoLayer


class LayerAggAct(nn.Module):
    def __init__(self, layerAggBinList, layerAggType, layerAct, channel, num_layer=3):
        super(LayerAggAct, self).__init__()
        self.layerAggBinList = layerAggBinList
        self.layerAggType = layerAggType
        if layerAggType > 1:
            self.LayerAgg = JumpingKnowledge(
                FEAT_FUSION_TYPE[layerAggType - 2], channel, num_layer
            )
        self.layerAggType = layerAggType
        if "sigmoid" == layerAct:
            self.Act = nn.Sigmoid()
        elif "tanh" == layerAct:
            self.Act = nn.Tanh()
        elif "relu" == layerAct:
            self.Act = nn.ReLU()
        elif "softplus" == layerAct:
            self.Act = nn.Softplus()
        elif "leaky_relu" == layerAct:
            self.Act = nn.LeakyReLU()
        elif "relu6" == layerAct:
            self.Act = nn.ReLU6()
        elif "elu" == layerAct:
            self.Act = nn.ELU()
        elif 0 == layerAct:
            self.Act = None
        else:
            raise NotImplementedError(f"Not implemented activation function {layerAct}")

    def forward(self, res):
        temp = []
        for i, value in enumerate(self.layerAggBinList):
            if value == 1:
                temp.append(res[i])
        if self.layerAggType == 0:
            x = res[0]
        elif self.layerAggType == 1:
            x = temp[0]
        elif self.layerAggType > 1:
            x = self.LayerAgg(temp)
        else:
            raise
        if self.Act is None:
            return x
        return self.Act(x)


class Network(nn.Module):
    def __init__(
        self,
        genotype,
        in_dim,
        out_dim,
        hidden_dim,
        dropout,
    ):
        super(Network, self).__init__()
        self.genoBin_sim = genotype
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.dropout = dropout

        self.reduction = Linear(self.in_dim, self.hidden_dim)
        self.hidden_layers = nn.ModuleList()
        self.classification = nn.ModuleList()
        self.num_layer = 0
        self.layer_aggregator = []
        self._construct_action()

    def _construct_action(self):
        index = 0
        if len(self.genoBin_sim) == 0:
            self.classification.append(LayerAggAct([], 0, 0, self.hidden_dim))
            self.classification.append(Linear(self.hidden_dim, self.out_dim))
        while index < len(self.genoBin_sim):
            layerAggBinList = self.genoBin_sim[index][0]
            layerAggType = self.genoBin_sim[index][1]
            assert layerAggType != 0
            if layerAggType == 1:
                assert sum(layerAggBinList) == 1

            layerAct = ACT_TYPE[self.genoBin_sim[index][2] - 1]
            if index < len(self.genoBin_sim) - 1:
                self.num_layer += 1
                nodeCC = CORR_COEFF_TYPE[self.genoBin_sim[index][3] - 1]
                nodeAgg = NODE_MSGAGG_TYPE[self.genoBin_sim[index][4] - 1]
                self.hidden_layers.append(
                    LayerAggAct(
                        layerAggBinList, layerAggType, layerAct, self.hidden_dim
                    )
                )
                hidden_dim = self._dim_cal(layerAggBinList, layerAggType)
                self.hidden_layers.append(
                    GeoLayer(
                        hidden_dim,
                        self.hidden_dim,
                        att_type=nodeCC,
                        agg_type=nodeAgg,
                        dropout=0.5,
                    )
                )
            else:
                self.classification.append(
                    LayerAggAct(
                        layerAggBinList, layerAggType, layerAct, self.hidden_dim
                    )
                )
                hidden_dim = self._dim_cal(layerAggBinList, layerAggType)
                self.classification.append(Linear(hidden_dim, self.out_dim))
            index += 1

    def forward(self, x, edge_index):
        x = self.reduction(x)
        res = [x]
        for i in range(self.num_layer):
            x = self.hidden_layers[i * 2](res)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.hidden_layers[i * 2 + 1](x, edge_index)
            res.append(x)
        x = self.classification[0](res)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classification[1](x)
        return x

    def _dim_cal(self, layerAggBinList, layerAggType):
        if layerAggType == 1:
            return self.hidden_dim
        else:
            layerAggType = FEAT_FUSION_TYPE[layerAggType - 2]
            if layerAggType == "cat":
                return sum(layerAggBinList) * self.hidden_dim
            else:
                return self.hidden_dim


class Individual(object):
    def __init__(self, max_len=8, genotype=None):
        self.max_len = max_len
        self.genoBin = genotype
        self.genoBin_sim = None
        self.genoDec = None
        self.val_loss = None
        self.test_acc = None
        self.instantiate()

    def instantiate(self):
        if self.genoBin is None:
            genoBin = []
            layerStatus = [1]
            for i in range(self.max_len):
                # Connections
                layerAggBinList = bin_arr_repair(
                    np.random.randint(0, 2, i + 1), layerStatus
                )
                # Fusion
                if sum(layerAggBinList) == 0:
                    genoBin.append([0, 0, 0, 0, 0])
                    layerStatus.append(0)
                    continue
                elif sum(layerAggBinList) == 1:
                    layerAggType = 1
                else:
                    layerAggType = random.randint(2, len(FEAT_FUSION_TYPE) + 1)
                # Activation
                layerAct = random.randint(1, len(ACT_TYPE))
                # Coefficients
                nodeCC = random.randint(1, len(CORR_COEFF_TYPE))
                # NodeAgg
                nodeAgg = random.randint(1, len(NODE_MSGAGG_TYPE))

                genoBin.append(
                    [layerAggBinList, layerAggType, layerAct, nodeCC, nodeAgg]
                )
                layerStatus.append(1)
            self.genoBin, self.genoBin_sim, self.genoDec = geno_repair(
                genoBin, self.max_len
            )
        else:
            layerStatus = [1]
            for index in range(self.max_len):
                if isinstance(self.genoBin[index][0], list):
                    self.genoBin[index][0] = bin_arr_repair(
                        self.genoBin[index][0], layerStatus
                    )
                    if sum(self.genoBin[index][0]) == 0:
                        self.genoBin[index] = [0, 0, 0, 0, 0]
                        layerStatus.append(0)
                        continue
                    elif sum(self.genoBin[index][0]) == 1:
                        self.genoBin[index][1] = 1
                    else:
                        if self.genoBin[index][1] <= 1:
                            self.genoBin[index][1] = random.randint(
                                2, len(FEAT_FUSION_TYPE) + 1
                            )
                    layerStatus.append(1)
                else:
                    layerStatus.append(0)
            self.genoBin, self.genoBin_sim, self.genoDec = geno_repair(
                self.genoBin, self.max_len
            )
