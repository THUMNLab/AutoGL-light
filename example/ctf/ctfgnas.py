import argparse
import numpy as np

from torch_geometric.datasets import Planetoid, Actor, WebKB

from autogllight.nas.space import CTFSpace
from autogllight.nas.estimator import OneShotOGBEstimator
from autogllight.nas.algorithm import CTF
from autogllight.utils.evaluation import Auc


def load_data(dataset_name: str):
    if dataset_name in ["Cora", "CiteSeer", "PubMed"]:
        dataset = Planetoid("./data/", dataset_name, split="full")
        data = dataset[0]
    elif dataset_name in ["Actor", "Cornell", "Texas", "Wisconsin"]:
        if dataset_name == "Actor":
            dataset = Actor("./data/Actor")
        else:
            dataset = WebKB("./data/", dataset_name)
        data = dataset[0]
        data.train_mask = data.train_mask[:, 0]
        data.val_mask = data.val_mask[:, 0]
        data.test_mask = data.test_mask[:, 0]
    else:
        raise NotImplementedError(f"The dataset {dataset_name} is not supported")
    num_node_features = dataset.num_node_features
    num_classes = dataset.num_classes
    return data, num_node_features, num_classes


def parse_args():
    parser = argparse.ArgumentParser(description="Implementation of CTFGNAS")
    # parameters for datasets
    parser.add_argument(
        "--data",
        type=str,
        default="Cora",
        help="Cora, CiteSeer, PubMed, Texas, Wisconsin, or Actor",
    )
    # parameters for EA
    parser.add_argument("--init_size", type=int, default=100)
    parser.add_argument("--pop_size", type=int, default=100)
    parser.add_argument("--max_gen", type=int, default=30)
    parser.add_argument("--max_len", type=int, default=8)
    parser.add_argument("--geno_len", type=int, default=5)
    parser.add_argument("--pc", type=float, default=0.9, help="Crossover probability")
    parser.add_argument("--pm", type=float, default=0.2, help="Mutation probability")
    parser.add_argument("--sn", type=float, default=30, help="surrogate number")
    parser.add_argument("--run_times", type=int, default=5)
    # parameters for training
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--wd", type=float, default=0.0005)
    parser.add_argument("--dropout", type=float, default=0.6)
    parser.add_argument(
        "--epochs", type=int, default=300, help="number of total epochs to run"
    )
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    # parameters for tuning
    # parser.add_argument("--tune_genotype", type=np.array, default=np.random.random())
    # parser.add_argument("--tune_epochs", type=int, default=300)
    # parser.add_argument("--tune_times", type=int, default=50)
    # parameters for log
    # parser.add_argument("--save", type=str, default="logs")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    data, num_node_features, num_classes = load_data(args.data)

    # TODO: use autogllight estimator

    space = CTFSpace(
        in_dim=num_node_features,
        out_dim=num_classes,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        pop_size=args.pop_size,
        max_len=args.max_len,
        args=args,
    )
    space.instantiate()
    algo = CTF(
        run_times=args.run_times,
        max_gen_num=args.max_gen,
        num_epochs=args.epochs,
        lr=args.lr,
        wd=args.wd,
        device="cuda",
    )
    algo.search(space, data, None)
