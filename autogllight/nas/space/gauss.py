from . import BaseSpace
from .gauss_space import COAUTHOR, RandomSampler, RLSampler, Supernet


class GaussSpace(BaseSpace):
    def __init__(self, input_dim, output_dim, add_pre, criterion, args):
        super().__init__()
        self.num_layers = args.num_layers
        self.n_input = input_dim  # data.x.size(-1)
        self.num_classes = output_dim  # num_classes, dataset.num_classes
        self.hidden_channels = args.hidden_channels
        self.dropout = args.dropout
        self.track = args.track
        self.add_pre = add_pre

        self.criterion = criterion
        self.args = args
        self.use_forward = True

    def build_graph(self):
        self.model = Supernet(
            self.num_layers,
            self.n_input,
            self.num_classes,
            self.hidden_channels,
            self.dropout,
            track=self.track,
            add_pre=self.add_pre,
        ).cuda()

        if self.args.use_sampler:
            self.sampler = RLSampler(
                COAUTHOR,
                self.args.num_layers,
                epochs=self.args.epoch_sampler,
                iter=self.args.iter_sampler,
                lr=self.args.lr_sampler,
                T=self.args.T,
                entropy=self.args.entropy,
            )
        else:
            self.sampler = RandomSampler(COAUTHOR, self.args.num_layers)

    def forward(self, data, arch):
        if not self.use_forward:
            return self.prediction

        pred = self.model(data, arch)

        self.current_pred = pred
        return pred
        # return pred0, pred, cosloss, sslout

    def keep_prediction(self):
        self.prediction = self.current_pred

    def parse_model(self, selection):
        self.use_forward = False
        return self.wrap()
