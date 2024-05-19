import random
import copy
import warnings

from .base import BaseSpace
from .ctf_space import *


class CTFSpace(BaseSpace):
    def __init__(
        self,
        # ? model config
        in_dim,
        out_dim,
        hidden_dim,
        dropout,
        # ? space config
        pop_size,
        max_len,
        geno_len,
        args,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.pop_size = pop_size
        self.max_len = max_len
        self.geno_len = geno_len

        self.init_size = args.init_size
        self.pc = args.pc
        self.pm = args.pm
        self.sn = args.sn

        self.to_be_eval = False

    def instantiate(self, **kwargs):
        """
        generate random distinct individuals
        """
        self.population = []
        self.genoDec_rec = set()
        while len(self.population) < self.init_size:
            indv = Individual(self.max_len)
            cur_genoDec = str(indv.genoDec)
            if cur_genoDec in self.genoDec_rec:
                continue
            self.population.append(indv)
            self.genoDec_rec.add(cur_genoDec)
        self.to_be_eval = True

    def build_graph(self, genotype):
        return Network(
            genotype, self.in_dim, self.out_dim, self.hidden_dim, self.dropout
        )

    def tune_graph(self, genotype, params):
        return Network(
            genotype, params.in_dim, params.out_dim, params.hidden_dim, params.dropout
        )

    def train(self):
        if self.to_be_eval:
            warnings.warn("The population has not been evaluated.")

        self.offspring = []
        while len(self.offspring) < self.pop_size:
            index1, individual1 = choose_parent(self.population)
            index2, individual2 = choose_parent(self.population)
            while index2 == index1:
                index2, individual2 = choose_parent(self.population)
            parent1 = copy.deepcopy(individual1.genoBin)
            parent2 = copy.deepcopy(individual2.genoBin)
            if random.random() < self.pc:
                position = random.randint(1, self.max_len - 1)
                child = [
                    parent1[:position] + parent2[position : self.max_len],
                    parent2[:position] + parent1[position : self.max_len],
                ]
                for genoBin in child:
                    layerStatus = [1]
                    for index in range(position):
                        if isinstance(genoBin[index][0], list):
                            layerStatus.append(1)
                        else:
                            layerStatus.append(0)
                    for index in range(position, self.max_len):
                        # The non-empty layer guarantees that it continues to be non-empty
                        if isinstance(genoBin[index][0], list):
                            # If the front connection is broken, reset it.
                            genoBin[index][0] = bin_arr_repair(
                                genoBin[index][0], layerStatus
                            )
                            while sum(genoBin[index][0]) == 0:
                                genoBin[index][0] = bin_arr_repair(
                                    np.random.randint(0, 2, index + 1), layerStatus
                                )
                            if sum(genoBin[index][0]) == 1:
                                genoBin[index][1] = 1
                            else:
                                if genoBin[index][1] == 1:
                                    genoBin[index][1] = random.randint(
                                        2, len(FEAT_FUSION_TYPE) + 1
                                    )
                            layerStatus.append(1)
                        else:
                            layerStatus.append(0)
                    # Make sure there are layers behind it to connect it
                    for index in range(position):
                        if isinstance(genoBin[index][0], list):
                            temp = 0
                            while temp == 0 and index < self.max_len - 1:
                                for j in range(index + 1, self.max_len):
                                    if isinstance(genoBin[j][0], list):
                                        temp += genoBin[j][0][index + 1]
                                if temp == 0:
                                    for j in range(index + 1, self.max_len):
                                        if isinstance(genoBin[j][0], list):
                                            genoBin[j][0][index + 1] = random.randint(
                                                0, 1
                                            )
                                            temp += genoBin[j][0][index + 1]
                                            if (
                                                sum(genoBin[j][0]) > 1
                                                and genoBin[j][1] == 1
                                            ):
                                                genoBin[j][1] = random.randint(
                                                    2, len(FEAT_FUSION_TYPE) + 1
                                                )
                                        else:
                                            if j == self.max_len - 1 and temp == 0:
                                                genoBin[j][0] = [
                                                    0 for _ in range(self.max_len)
                                                ]
                                                genoBin[j][0][index + 1] = 1
                                                genoBin[j][1] = 1
                                                genoBin[j][2] = random.randint(
                                                    1, len(ACT_TYPE)
                                                )
            else:
                child = [parent1, parent2]
            for genoBin in child:
                genoBin = self._mutate_repair(genoBin)
                individual = Individual(self.max_len, genoBin)
                if len(self.offspring) < self.pop_size:
                    self.offspring.append(individual)
                else:
                    break

        self.to_be_eval = True

    def _mutate_repair(self, genoBin0):
        genoBin = copy.deepcopy(genoBin0)
        layerStatus = [1]
        for i in range(self.max_len):  # the i-th layer
            if random.random() < self.pm:
                if isinstance(genoBin[i][0], list):  # non-empty layer
                    state = True
                    while state is True:
                        # Connection, Fusion, Activation, Coefficients, NodeAgg
                        mutation_type = random.randint(0, self.geno_len - 1)
                        if mutation_type == 0 and i < self.max_len - 1:
                            layerAggBinList = bin_arr_repair(
                                np.random.randint(0, 2, i + 1), layerStatus
                            )
                            while layerAggBinList == genoBin[i][0]:
                                layerAggBinList = bin_arr_repair(
                                    np.random.randint(0, 2, i + 1), layerStatus
                                )
                            genoBin[i][0] = layerAggBinList
                            if sum(genoBin[i][0]) == 0 and i < self.max_len - 1:
                                genoBin[i] = [0, 0, 0, 0, 0]
                                state = False
                                layerStatus_temp = copy.deepcopy(layerStatus)
                                layerStatus_temp.append(0)
                                for index in range(i + 1, self.max_len):
                                    # The non-empty layer guarantees that it continues to be non-empty
                                    if isinstance(genoBin[index][0], list):
                                        genoBin[index][0] = bin_arr_repair(
                                            genoBin[index][0], layerStatus_temp
                                        )
                                        while sum(genoBin[index][0]) == 0:
                                            genoBin[index][0] = bin_arr_repair(
                                                np.random.randint(0, 2, index + 1),
                                                layerStatus_temp,
                                            )
                                        if sum(genoBin[index][0]) == 1:
                                            genoBin[index][1] = 1
                                        else:
                                            if genoBin[index][1] == 1:
                                                genoBin[index][1] = random.randint(
                                                    2, len(FEAT_FUSION_TYPE) + 1
                                                )
                                        layerStatus_temp.append(1)
                                    else:
                                        layerStatus_temp.append(0)
                                # Make sure there are layers behind it to connect it
                                for index in range(i):
                                    if isinstance(genoBin[index][0], list):
                                        temp = 0
                                        while temp == 0 and index < self.max_len - 1:
                                            for j in range(index + 1, self.max_len):
                                                if isinstance(genoBin[j][0], list):
                                                    temp += genoBin[j][0][index + 1]
                                            if temp == 0:
                                                for j in range(index + 1, self.max_len):
                                                    if isinstance(genoBin[j][0], list):
                                                        genoBin[j][0][index + 1] = (
                                                            random.randint(0, 1)
                                                        )
                                                        temp += genoBin[j][0][index + 1]
                                                        if (
                                                            sum(genoBin[j][0]) > 1
                                                            and genoBin[j][1] == 1
                                                        ):
                                                            genoBin[j][1] = (
                                                                random.randint(
                                                                    2,
                                                                    len(
                                                                        FEAT_FUSION_TYPE
                                                                    )
                                                                    + 1,
                                                                )
                                                            )
                                                    else:
                                                        if (
                                                            j == self.max_len - 1
                                                            and temp == 0
                                                        ):
                                                            genoBin[j][0] = [
                                                                0
                                                                for _ in range(
                                                                    self.max_len
                                                                )
                                                            ]
                                                            genoBin[j][0][i + 1] = 1
                                                            genoBin[j][1] = 1
                                                            genoBin[j][2] = (
                                                                random.randint(
                                                                    1,
                                                                    len(ACT_TYPE),
                                                                )
                                                            )
                            elif sum(genoBin[i][0]) == 1:
                                genoBin[i][1] = 1
                                state = False
                            else:
                                if genoBin[i][1] == 1:
                                    genoBin[i][1] = random.randint(
                                        2, len(FEAT_FUSION_TYPE) + 1
                                    )
                                    state = False
                        elif mutation_type == 1:
                            if genoBin[i][1] > 1:
                                genoBin[i][1] = get_distinct_randint(
                                    2, len(FEAT_FUSION_TYPE) + 1, genoBin[i][1]
                                )
                                state = False
                            elif genoBin[i][1] == 1:
                                state = True
                            else:
                                raise
                        elif mutation_type == 2:
                            genoBin[i][2] = get_distinct_randint(
                                1, len(ACT_TYPE), genoBin[i][2]
                            )
                            state = False
                        elif mutation_type == 3:
                            if i < self.max_len - 1:
                                genoBin[i][3] = get_distinct_randint(
                                    1, len(CORR_COEFF_TYPE), genoBin[i][3]
                                )
                                state = False
                        elif mutation_type == 4:
                            if i < self.max_len - 1:
                                genoBin[i][4] = get_distinct_randint(
                                    1, len(NODE_MSGAGG_TYPE), genoBin[i][4]
                                )
                                state = False
                # empty layer
                else:
                    while genoBin[i][0] == 0 or sum(genoBin[i][0]) == 0:
                        # Connections
                        layerAggBinList = bin_arr_repair(
                            np.random.randint(0, 2, i + 1), layerStatus
                        )
                        # Fusion
                        if sum(layerAggBinList) == 0:
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

                        if i < self.max_len - 1:
                            genoBin[i] = [
                                layerAggBinList,
                                layerAggType,
                                layerAct,
                                nodeCC,
                                nodeAgg,
                            ]
                        else:
                            genoBin[i] = [layerAggBinList, layerAggType, layerAct]
                        temp = 0
                        while temp == 0 and i < self.max_len - 1:
                            for j in range(i + 1, self.max_len):
                                if isinstance(genoBin[j][0], list):
                                    genoBin[j][0][i + 1] = random.randint(0, 1)
                                    temp += genoBin[j][0][i + 1]
                                    if sum(genoBin[j][0]) > 1 and genoBin[j][1] == 1:
                                        genoBin[j][1] = random.randint(
                                            2, len(FEAT_FUSION_TYPE) + 1
                                        )
                                else:
                                    if j == self.max_len - 1 and temp == 0:
                                        genoBin[j][0] = [0 for _ in range(self.max_len)]
                                        genoBin[j][0][i + 1] = 1
                                        genoBin[j][1] = 1
                                        genoBin[j][2] = random.randint(1, len(ACT_TYPE))
            if isinstance(genoBin[i][0], list):
                layerStatus.append(1)
            else:
                layerStatus.append(0)
        return genoBin

    def _select_index(self):
        random_range = []
        selected_length = int(self.sn * 2)
        while len(random_range) < selected_length:
            num = random.randint(0, self.pop_size - 1)
            if num not in random_range:
                random_range.append(num)
        selected_index = []
        for i in range(0, selected_length, 2):
            if (
                self.offspring[random_range[i]].val_loss
                <= self.offspring[random_range[i + 1]].val_loss
            ):
                selected_index.append(random_range[i])
            else:
                selected_index.append(random_range[i + 1])
        return selected_index

    def forward(self):
        populations = self.population + self.offspring
        random_range = []
        selected_length = int(self.pop_size * 2)
        while len(random_range) < selected_length:
            num = random.randint(0, selected_length - 1)
            if num not in random_range:
                random_range.append(num)
        selected_index = []
        for i in range(0, selected_length, 2):
            if (
                populations[random_range[i]].val_loss
                <= populations[random_range[i + 1]].val_loss
            ):
                selected_index.append(random_range[i])
            else:
                selected_index.append(random_range[i + 1])
        self.population, pop_objs = [], []
        for i in selected_index:
            self.population.append(populations[i])
            pop_objs.append(populations[i].val_loss)
        self.offspring = []

        self.to_be_eval = False
