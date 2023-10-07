"""This file is from rationale"""
from math import degrees
import torch
import numpy as np
import torch_geometric

class GroupAssigner(object):
    def __init__(self, criterion=None, n_groups=2, prob=None):
        self.criterion = criterion # func(data) -> group assignment
        self.n_groups = n_groups
        if self.criterion is None:
            self.prob = prob if prob else torch.ones(n_groups).float()/n_groups

    def __call__(self, data):
        if self.criterion is not None:
            group_id = self.criterion(data)
        else:
            group_id = torch.tensor(
                np.random.choice(range(self.n_groups), 1, p=self.prob)
                ).long()
        data.group_id = group_id
        return data

class DegreeDistribution(object):

    def __call__(self, g):
        '''if g.is_undirected():
            edges = g.edge_index[0]
        else:
            edges = torch.cat((g.edge_index[0], g.edge_index[1]))'''
        edges = g.edge_index[1]
        if edges.numel() == 0:
            deratio = torch.tensor([0.0, 0.0, 0.0])
        else:
            degrees = torch_geometric.utils.degree(edges).to(torch.long).numpy().tolist()
            deratio = [degrees.count(i) for i in range(1, 4)]
            deratio = torch.tensor(deratio) / g.num_nodes
        g.deratio = deratio
        return g
