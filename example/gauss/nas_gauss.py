import argparse
import os
import sys

import torch
from autogllight.nas.algorithm import Gauss
from autogllight.nas.estimator import OneShotOGBEstimator
from autogllight.nas.space import GaussSpace
from autogllight.utils import set_seed
from autogllight.utils.evaluation import Auc

from torch_geometric.datasets import Coauthor

sys.path.append("..")
sys.path.append(".")
os.environ["AUTOGL_BACKEND"] = "pyg"


def parser_args():
    parser = argparse.ArgumentParser(description='gen_models')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dataset', type=str, default='CS', choices=['CS', 'Physics'])

    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--eval-epoch', type=int, default=-1)
    parser.add_argument('--save-epoch', type=int, default=5)
    parser.add_argument('--wd', type=float, default=0.0)
    parser.add_argument('--repeat', type=int, default=10)
    parser.add_argument('--track', action='store_true')
    parser.add_argument('--space', type=str, default='simple', choices=['simple', 'full'])
    parser.add_argument('--name', type=str, default='search')
    parser.add_argument('--restore-epoch', type=int, default=-1)
    parser.add_argument('--restore-folder', type=str)

    # arch sampler
    parser.add_argument('--use-sampler', action='store_true')
    parser.add_argument("--sampler-fit", type=int, default=5)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--no-baseline", action="store_true", help="whether to force multiplying 1.")
    parser.add_argument("--restart", type=int, default=-1)
    parser.add_argument("--warm-up", type=float, default=0.4)
    parser.add_argument("--lr-sampler", type=float, default=1e-3)
    parser.add_argument("--epoch-sampler", type=int, default=5)
    parser.add_argument("--iter-sampler", type=int, default=7)
    parser.add_argument("--entropy", type=float, default=0.0)
    parser.add_argument("--max-clip", type=float, default=-1)
    parser.add_argument('--min-clip', type=float, default=-1)

    # curriculum
    parser.add_argument('--use-curriculum', action='store_true')
    parser.add_argument("--max-ratio", type=float, default=0.2)
    parser.add_argument("--min-ratio", type=float, nargs="+", default=[0.2, 1.0])


    return args


if __name__ == "__main__":
    set_seed(0)
    hps = {
        "eval_epoch": -1,
        "epochs": 100,
        "use_sampler": True,
        "warm_up": 0.6,
        "T": 100,
        "use_curriculum": True,
        "name": "T100",
        "dataset": "CS",
    }
    args = parser_args()

    for k, v in hps.items():
        setattr(args, k, v)
    args.warm_up = int(args.warm_up * args.epochs)

    dataset = Coauthor(os.path.expanduser("~/dataset/pyg"), name=args.dataset, transform=T.NormalizeFeatures())
    data = dataset[0]
    num_features = data.x.size(-1)
    num_classes = dataset.num_classes

    space = GaussSpace(
        input_dim=num_features,
        output_dim=num_classes,
        hidden_channels=args.hidden_channels,
        add_pre=(args.dataset == 'Physics'),
        args=args,
    )


    # estimator = OneShotOGBEstimator(
    #     loss_f="binary_cross_entropy_with_logits", evaluation=[Auc()]
    # )


    space.instantiate()

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    algo = Gauss(num_epochs=args.epochs, device=device, args=args)
    algo.search(space, data, estimator)