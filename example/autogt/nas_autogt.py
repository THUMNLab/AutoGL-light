from argparse import ArgumentParser
import os
import sys

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from autogllight.nas.algorithm import Autogt
from autogllight.nas.estimator import OneShotEstimator
from autogllight.nas.space import AutogtSpace
from autogllight.utils import set_seed

from data import get_dataset

sys.path.append("..")
sys.path.append(".")
os.environ["AUTOGL_BACKEND"] = "pyg"


def parser_args():
    parent_parser = ArgumentParser()
    parser = parent_parser.add_argument_group("GraphTransformer")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--ffn_dim', type=int, default=32)
    parser.add_argument('--intput_dropout_rate', type=float, default=0.1)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--attention_dropout_rate',type=float, default=0.1)
    parser.add_argument('--checkpoint_path', type=str, default='')
    parser.add_argument('--warmup_updates', type=int, default=600)
    parser.add_argument('--tot_updates', type=int, default=5000)
    parser.add_argument('--peak_lr', type=float, default=2e-4)
    parser.add_argument('--end_lr', type=float, default=1e-9)
    parser.add_argument('--edge_type', type=str, default='multi_hop')
    parser.add_argument('--validate', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--dataset_name', type=str, default='PROTEINS')
    parser.add_argument('--multi_hop_max_dist', type=int, default=5)
    parser.add_argument('--num_class', type=int, default=1)
    parser.add_argument('--path', type=str, default='/home/zeyuan.yin/My-AutoGL/paper/AutoGT/json/PROTEINS.json')

    parser = parent_parser.add_argument_group("Dataset")
    parser.add_argument('--lap_enc_dim', type=int, default=10)
    parser.add_argument('--svd_enc_dim', type=int, default=16)
    parser.add_argument('--pma_dim', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_node', type=int, default=512)
    parser.add_argument('--data_split', type=int, default=0)

    parser = parent_parser.add_argument_group("Training")
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--split_epochs', type=int, default=60)
    parser.add_argument('--end_epochs', type=int, default=200)
    parser.add_argument('--retrain_epochs', type=int, default=200)

    parser = parent_parser.add_argument_group("Evolution")
    parser.add_argument('--evol_epochs', type=int, default=20)
    parser.add_argument('--population_num', type=int, default=100)
    parser.add_argument('--m_prob', type=float, default=0.3)
    parser.add_argument('--mutation_num', type=int, default=20)
    parser.add_argument('--hybridization_num', type=int, default=20)
    parser.add_argument('--retrain_num', type=int, default=4)

    parser.add_argument('--device', type=int, default=3)

    return parent_parser.parse_args()


class MyOneShotEstimator(OneShotEstimator):
    def infer(self, model, dataset, arch, mask):
        device = next(model.parameters()).device
        dataset = dataset.to(device)

        pred = model(dataset, arch)[mask]
        y = data.y[mask]

        loss = getattr(F, self.loss_f)(pred, y)
        probs = F.softmax(pred, dim=1).detach().cpu().numpy()
        y = y.cpu()
        metrics = {
            eva.get_eval_name(): eva.evaluate(probs, y) for eva in self.evaluation
        }
        return metrics, loss

if __name__ == "__main__":
    set_seed(0)
    hps = {
        "batch_size": 48,
        "max_node": 512,
        "split_epochs": 50,
        "end_epochs":200,
        "warmup_updates": 600,
        "tot_updates": 5000,
        "n_layers": 4,
        "num_heads": 4,
        "hidden_dim": 32,
        "ffn_dim": 32,
    }
    args = parser_args()

    for k, v in hps.items():
        setattr(args, k, v)

    space = AutogtSpace(
        args=args,
    )

    space.instantiate()


    estimator = MyOneShotEstimator()

    # device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # x = torch.Tensor([1, 2, 3])

    # x = x.to(device)
    # print(x)
    # print(device)
    # exit()
    device = torch.device(device)

    algo = Autogt(num_epochs=args.epochs, device=device, args=args)



    data = get_dataset(args, args.data_split)

    algo.search(space, data, estimator)
