import os

os.environ["AUTOGL_BACKEND"] = "pyg"
import yaml
import random
import numpy as np
from autogllight.utils import *
from autogllight.nas.space import AutoAttendNodeClassificationSpace
from autogllight.nas.algorithm import Spos
from autogllight.nas.estimator import OneShotEstimator
from torch_geometric.datasets import Planetoid
from os import path as osp
import torch_geometric.transforms as T


def test_autoattend():
    dataname = "cora"
    dataset = Planetoid(
        osp.expanduser("~/.cache-autogl"), dataname, transform=T.NormalizeFeatures()
    )
    data = dataset[0]
    label = data.y
    input_dim = data.x.shape[-1]
    num_classes = len(np.unique(label.numpy()))

    space = AutoAttendNodeClassificationSpace(
        input_dim=input_dim, output_dim=num_classes
    )
    space.instantiate()
    algo = Spos(n_warmup=10, cycles=200)
    estimator = OneShotEstimator()
    algo.search(space, dataset, estimator)
