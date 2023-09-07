from . import BaseSpace
from .gauss_space import *


class GaussSpace(BaseSpace):
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
        self.supernet0 = Supernet(args.num_layers, data.x.size(-1), dataset.num_classes, args.hidden_channels, args.dropout, track=args.track, add_pre=add_pre).cuda()

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
