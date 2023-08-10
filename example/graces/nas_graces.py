import argparse
import os
import random
import sys
from os import path as osp

import numpy as np
import torch_geometric.transforms as T
import yaml
from graces_dataset import load_data
from torch_geometric.datasets import Planetoid
from tqdm import trange

from autogllight.nas.algorithm import Gasso, Graces, RandomSearch
from autogllight.nas.estimator import OneShotEstimator, TrainScratchEstimator
from autogllight.nas.space import (
    GassoSpace,
    GracesSpace,
    SinglePathNodeClassificationSpace,
)
from autogllight.utils import *
# from autogllight.utils.evaluation import Auc_ogb

sys.path.append("..")
sys.path.append(".")
os.environ["AUTOGL_BACKEND"] = "pyg"
# OgbEstimator,


# Graces


def parser_args():
    graph_classification_dataset = [
        "DD",
        "MUTAG",
        "PROTEINS",
        "NCI1",
        "NCI109",
        "IMDB-BINARY",
        "REDDIT-BINARY",
        "BZR",
        "COX2",
        "IMDB-MULTI",
        "COLORS-3",
        "COLLAB",
        "REDDIT-MULTI-5K",
        "synthetic",
        "spmotif",
        "ogbg-molbbbp",
        "ogbg-molsider",
        "ogbg-molhiv",
        "ogbg-moltox21",
        "ogbg-molclintox",
        "ogbg-molbace",
        "ogbg-moltoxcast",
    ]
    node_classification_dataset = [
        "Cora",
        "CiteSeer",
        "PubMed",
        "Amazon_Computers",
        "Coauthor_CS",
        "Coauthor_Physics",
        "Amazon_Photo",
        "small_Reddit",
        "small_arxiv",
        "Reddit",
        "ogbn-arxiv",
    ]
    parser = argparse.ArgumentParser("pas-train-search")
    parser.add_argument(
        "--data", type=str, default="ogbg-molbace", help="location of the data corpus"
    )
    parser.add_argument(
        "--record_time",
        action="store_true",
        default=False,
        help="used for run_with_record_time func",
    )
    parser.add_argument("--batch_size", type=int, default=512, help="batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=0.01, help="init learning rate"
    )
    parser.add_argument(
        "--learning_rate_min", type=float, default=0.001, help="min learning rate"
    )
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="weight decay")
    parser.add_argument("--gpu", type=int, default=3, help="gpu device id")
    parser.add_argument(
        "--epochs", type=int, default=100, help="num of training epochs"
    )
    parser.add_argument(
        "--model_path", type=str, default="saved_models", help="path to save the model"
    )
    parser.add_argument("--save", type=str, default="EXP", help="experiment name")
    parser.add_argument(
        "--save_file", action="store_true", default=False, help="save the script"
    )
    parser.add_argument("--seed", type=int, default=2, help="random seed")
    parser.add_argument("--grad_clip", type=float, default=5, help="gradient clipping")
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.0,
        help="the explore rate in the gradient descent process",
    )
    parser.add_argument(
        "--train_portion", type=float, default=0.5, help="portion of training data"
    )
    parser.add_argument(
        "--unrolled",
        action="store_true",
        default=False,
        help="use one-step unrolled validation loss",
    )
    parser.add_argument(
        "--temperature", type=float, default=1, help="temperature of AGLayer"
    )
    parser.add_argument(
        "--arch_learning_rate",
        type=float,
        default=0.08,
        help="learning rate for arch encoding",
    )
    # parser.add_argument('--arch_learning_rate', type=float, default=0.08, help='learning rate for arch encoding')
    parser.add_argument(
        "--arch_learning_rate_min",
        type=float,
        default=0.0,
        help="minimum learning rate for arch encoding",
    )
    # parser.add_argument('--cos_arch_lr', action='store_true', default=False, help='lr decay for learning rate')
    parser.add_argument(
        "--arch_weight_decay",
        type=float,
        default=1e-3,
        help="weight decay for arch encoding",
    )
    parser.add_argument(
        "--gnn0_learning_rate",
        type=float,
        default=0.005,
        help="learning rate for arch encoding",
    )
    parser.add_argument(
        "--gnn0_learning_rate_min",
        type=float,
        default=0.0,
        help="minimum learning rate for arch encoding",
    )
    parser.add_argument(
        "--gnn0_weight_decay",
        type=float,
        default=1e-3,
        help="weight decay for arch encoding",
    )
    parser.add_argument(
        "--pooling_ratio", type=float, default=0.5, help="global pooling ratio"
    )
    parser.add_argument("--beta", type=float, default=5e-3, help="global pooling ratio")
    parser.add_argument("--gamma", type=float, default=5.0, help="global pooling ratio")
    parser.add_argument("--eta", type=float, default=0.1, help="global pooling ratio")
    parser.add_argument(
        "--eta_max", type=float, default=0.5, help="global pooling ratio"
    )
    parser.add_argument(
        "--with_conv_linear",
        type=bool,
        default=False,
        help=" in NAMixOp with linear op",
    )
    parser.add_argument(
        "--num_layers", type=int, default=3, help="num of layers of GNN method."
    )
    parser.add_argument(
        "--withoutjk", action="store_true", default=False, help="remove la aggregtor"
    )
    parser.add_argument(
        "--alpha_mode",
        type=str,
        default="train_loss",
        help="how to update alpha",
        choices=["train_loss", "valid_loss", "valid_acc"],
    )
    parser.add_argument(
        "--search_act",
        action="store_true",
        default=False,
        help="search act in supernet.",
    )
    parser.add_argument(
        "--hidden_size", type=int, default=128, help="default hidden_size in supernet"
    )
    parser.add_argument(
        "--BN", type=int, default=64, help="default hidden_size in supernet"
    )
    parser.add_argument(
        "--graph_dim", type=int, default=8, help="default hidden_size in supernet"
    )
    # parser.add_argument('--attention_dim',  type=int, default=16, help='default hidden_size in supernet')
    # parser.add_argument('--layer_emb_dim',  type=int, default=16, help='default hidden_size in supernet')
    # parser.add_argument('--op_dim',  type=int, default=16, help='default hidden_size in supernet')
    parser.add_argument(
        "--attention_dropout",
        type=float,
        default=0.1,
        help="default hidden_size in supernet",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.5, help="default hidden_size in supernet"
    )
    parser.add_argument(
        "--num_sampled_archs", type=int, default=5, help="sample archs from supernet"
    )

    # for ablation stuty
    parser.add_argument(
        "--remove_pooling",
        action="store_true",
        default=False,
        help="remove pooling block.",
    )
    parser.add_argument(
        "--remove_readout",
        action="store_true",
        default=False,
        help="exp5, only search the last readout block.",
    )
    parser.add_argument(
        "--remove_jk",
        action="store_true",
        default=False,
        help="remove ensemble block, Graph representation = Z3",
    )

    # in the stage of update theta.
    parser.add_argument(
        "--temp", type=float, default=0.2, help=" temperature in gumble softmax."
    )
    parser.add_argument(
        "--loc_mean",
        type=float,
        default=10.0,
        help="initial mean value to generate the location",
    )
    parser.add_argument(
        "--loc_std",
        type=float,
        default=0.01,
        help="initial std to generate the location",
    )
    parser.add_argument(
        "--lamda",
        type=int,
        default=2,
        help="sample lamda architectures in calculate natural policy gradient.",
    )
    parser.add_argument(
        "--adapt_delta",
        action="store_true",
        default=False,
        help="adaptive delta in update theta.",
    )
    parser.add_argument(
        "--delta", type=float, default=1.0, help="a fixed delta in update theta."
    )
    parser.add_argument(
        "--w_update_epoch", type=int, default=1, help="epoches in update W"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="darts",
        help="how to update alpha",
        choices=["mads", "darts", "snas"],
    )

    args = parser.parse_args()
    args.graph_classification_dataset = graph_classification_dataset
    args.node_classification_dataset = node_classification_dataset
    torch.set_printoptions(precision=4)

    return args


if __name__ == "__main__":
    set_seed(0)

    hps = {
        "num_layers": 2,
        "learning_rate": 0.00034828472005404485,
        "learning_rate_min": 0.00019242101475226765,
        "weight_decay": 0,
        "temperature": 4.089492969843236,
        "arch_learning_rate": 0.0003948218327378405,
        "arch_weight_decay": 0.001,
        "gnn0_learning_rate": 0.03391343886431106,
        "gnn0_weight_decay": 0,
        "pooling_ratio": 0.3029329352563719,
        "dropout": 0.2623320360058418,
        "beta": 0.00462003423626971,
        "eta": 0.06839360891312493,
        "eta_max": 0.5871192734433861,
        "gamma": 0.010494136340498105,
    }
    args = parser_args()

    for k, v in hps.items():
        setattr(args, k, v)

    data, num_nodes = load_data(
        args.data, batch_size=args.batch_size, split_seed=args.seed
    )

    num_features = data[0].num_features
    num_classes = data[0].num_tasks

    criterion = torch.nn.BCEWithLogitsLoss()

    from ogb.graphproppred import Evaluator

    estimator_ogb = Evaluator("ogbg-molbace")
    # estimator = OneShotEstimator(evaluation=Auc_ogb)

    space = GracesSpace(
        input_dim=num_features,
        output_dim=num_classes,
        num_nodes=num_nodes,
        mol=True,  # for ogbg
        virtual=True,  # for ogbg
        criterion=torch.nn.BCEWithLogitsLoss(),  # for ogbg
        args=args,
    )

    space.instantiate()
    # print(space.model)
    # exit()

    algo = Graces(num_epochs=args.epochs, args=args)

    algo.search(space, data, estimator_ogb)
    # algo.search(space, data, estimator)
