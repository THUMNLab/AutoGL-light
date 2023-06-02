import os

os.environ["AUTOGL_BACKEND"] = "pyg"
import yaml
import random
import numpy as np
from autogllight.utils import *
from autogllight.nas.space import SinglePathNodeClassificationSpace
from autogllight.nas.algorithm import RandomSearch, Darts, RL, GraphNasRL, Enas
from autogllight.nas.estimator import OneShotEstimator
from torch_geometric.datasets import Planetoid
from os import path as osp
import torch_geometric.transforms as T


def get_default():
    dataname = "cora"
    dataset = Planetoid(
        osp.expanduser("~/.cache-autogl"), dataname, transform=T.NormalizeFeatures()
    )
    data = dataset[0]
    label = data.y
    input_dim = data.x.shape[-1]
    num_classes = len(np.unique(label.numpy()))

    space = SinglePathNodeClassificationSpace(
        input_dim=input_dim, output_dim=num_classes
    )
    space.instantiate()
    return dataset, space


def test_random_search():
    set_seed(0)
    algo = RandomSearch(num_epochs=2)
    estimator = OneShotEstimator()
    dataset, space = get_default()
    algo.search(space, dataset, estimator)


def test_darts():
    set_seed(0)
    algo = Darts(num_epochs=100)
    estimator = OneShotEstimator()
    dataset, space = get_default()
    algo.search(space, dataset, estimator)


def test_rl_ws():
    set_seed(0)
    algo = RL(num_epochs=2, ctrl_steps_aggregate=2, weight_share=True)
    estimator = OneShotEstimator()
    dataset, space = get_default()
    algo.search(space, dataset, estimator)


def test_rl_nws():
    set_seed(0)
    algo = RL(num_epochs=2, ctrl_steps_aggregate=2, weight_share=False)
    estimator = OneShotEstimator()
    dataset, space = get_default()
    algo.search(space, dataset, estimator)


def test_graphnas_rl_ws():
    set_seed(0)
    algo = GraphNasRL(num_epochs=2, ctrl_steps_aggregate=2, weight_share=True)
    estimator = OneShotEstimator()
    dataset, space = get_default()
    algo.search(space, dataset, estimator)


def test_graphnas_rl_nws():
    set_seed(0)
    algo = GraphNasRL(num_epochs=2, ctrl_steps_aggregate=2, weight_share=False)
    estimator = OneShotEstimator()
    dataset, space = get_default()
    algo.search(space, dataset, estimator)


def test_enas():
    set_seed(0)
    algo = Enas(num_epochs=2, ctrl_steps_aggregate=2)
    estimator = OneShotEstimator()
    dataset, space = get_default()
    algo.search(space, dataset, estimator)
