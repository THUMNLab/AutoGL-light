import os

os.environ["AUTOGL_BACKEND"] = "pyg"
import yaml
import random
import numpy as np
from autogllight.utils import *
from autogllight.nas.space import (
    GraphNasNodeClassificationSpace,
    GraphNasMacroNodeClassificationSpace,
)
from autogllight.nas.algorithm import GraphNasRL
from autogllight.nas.estimator import OneShotEstimator
from torch_geometric.datasets import Planetoid
from os import path as osp
import torch_geometric.transforms as T


def test_graphnas():
    dataname = "cora"
    dataset = Planetoid(
        osp.expanduser("~/.cache-autogl"), dataname, transform=T.NormalizeFeatures()
    )
    data = dataset[0]
    label = data.y
    input_dim = data.x.shape[-1]
    num_classes = len(np.unique(label.numpy()))

    space = GraphNasNodeClassificationSpace(input_dim=input_dim, output_dim=num_classes)
    space.instantiate()
    algo = GraphNasRL(num_epochs=2, ctrl_steps_aggregate=2, weight_share=False)
    estimator = OneShotEstimator()
    algo.search(space, dataset, estimator)


def test_graphnas_macro():
    dataname = "cora"
    dataset = Planetoid(
        osp.expanduser("~/.cache-autogl"), dataname, transform=T.NormalizeFeatures()
    )
    data = dataset[0]
    label = data.y
    input_dim = data.x.shape[-1]
    num_classes = len(np.unique(label.numpy()))

    space = GraphNasMacroNodeClassificationSpace(
        input_dim=input_dim, output_dim=num_classes
    )
    space.instantiate()
    algo = GraphNasRL(num_epochs=2, ctrl_steps_aggregate=2, weight_share=False)
    estimator = OneShotEstimator()
    algo.search(space, dataset, estimator)
