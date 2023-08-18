from . import BaseSpace
from .graces_space import *


class GracesSpace(BaseSpace):
    def __init__(self, input_dim, output_dim, num_nodes, mol, virtual, criterion, args):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_nodes = num_nodes
        self.mol = mol
        self.virtual = virtual
        self.criterion = criterion
        self.args = args
        self.use_forward = True

    def build_graph(self):
        self.supernet0 = GEncoder(
            criterion=self.criterion,
            in_dim=self.input_dim,
            out_dim=self.output_dim,
            hidden_size=self.args.graph_dim,
            num_layers=2,
            dropout=0.5,
            epsilon=self.args.epsilon,
            args=self.args,
            with_conv_linear=self.args.with_conv_linear,
            num_nodes=self.num_nodes,
            mol=self.mol,
            virtual=self.virtual,
        )
        self.supernet = Network(
            criterion=self.criterion,
            in_dim=self.input_dim,
            out_dim=self.output_dim,
            hidden_size=self.args.hidden_size,
            num_layers=self.args.num_layers,
            dropout=self.args.dropout,
            epsilon=self.args.epsilon,
            args=self.args,
            with_conv_linear=self.args.with_conv_linear,
            num_nodes=self.num_nodes,
            mol=self.mol,
            virtual=self.virtual,
        )
        num_na_ops = len(NA_PRIMITIVES)
        num_pool_ops = len(POOL_PRIMITIVES)
        self.ag = AG(args=self.args, num_op=num_na_ops, num_pool=num_pool_ops)
        self.explore_num = 0

    def forward(self, data):
        if not self.use_forward:
            return self.prediction
        # mhssl
        pred0, graph_emb, sslout = self.supernet0(data, mode="mixed")
        # ag
        graph_alpha, cosloss = self.ag(graph_emb)
        # final supernet
        pred, _ = self.supernet(data, mode="mads", graph_alpha=graph_alpha)
        self.current_pred = pred
        return pred0, pred, cosloss, sslout

    def keep_prediction(self):
        self.prediction = self.current_pred

    def parse_model(self, selection):
        self.use_forward = False
        return self.wrap()
