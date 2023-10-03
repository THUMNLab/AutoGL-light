import torch

from . import BaseSpace
from .autogt_space import GraphTransformer


class AutogtSpace(BaseSpace):
    def __init__(self, args):
        super().__init__()
        # self.num_layers = args.num_layers
        # self.input_dim = input_dim  # data.x.size(-1)
        # self.output_dim = output_dim  # num_classes, dataset.num_classes
        # self.num_classes = output_dim
        # self.hidden_channels = args.hidden_channels
        # self.dropout = args.dropout
        # self.track = args.track
        # self.add_pre = add_pre
        self.args = args
        self.use_forward = True

    def build_graph(self):
        self.model = GraphTransformer(
            n_layers=self.args.n_layers,
            num_heads=self.args.num_heads,
            hidden_dim=self.args.hidden_dim,
            attention_dropout_rate=self.args.attention_dropout_rate,
            num_class=self.args.num_class,
            dropout_rate=self.args.dropout_rate,
            intput_dropout_rate=self.args.intput_dropout_rate,
            weight_decay=self.args.weight_decay,
            ffn_dim=self.args.ffn_dim,
            dataset_name=self.args.dataset_name,
            warmup_updates=self.args.warmup_updates,
            tot_updates=self.args.tot_updates,
            peak_lr=self.args.peak_lr,
            end_lr=self.args.end_lr,
            edge_type=self.args.edge_type,
            multi_hop_max_dist=self.args.multi_hop_max_dist,
            lap_dim=self.args.lap_enc_dim,
            svd_dim=self.args.svd_enc_dim,
            path=self.args.path,
        ).cuda()


    def load_model(self, path):
        self.build_graph()
        optimizer, lr_scheduler = self.model.configure_optimizers()
        scheduler = lr_scheduler['scheduler']
        info = torch.load(path)
        self.model.load_state_dict(info['model'])
        optimizer.load_state_dict(info['optimizer'])
        scheduler.load_state_dict(info['scheduler'])
        print("Load Successfully!")
        return self.model, optimizer, scheduler


    def save_model(self, optimizer, scheduler, path):
        print("Saving Model to Path: " + path)
        torch.save({'model': self.model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}, path)
        print("Save Successfully!")



    def forward(self, data, params):
        if not self.use_forward:
            return self.prediction

        pred = self.model(data, params)
        self.current_pred = pred
        return pred

    def keep_prediction(self):
        self.prediction = self.current_pred

    def parse_model(self, selection):
        self.use_forward = False
        return self.wrap()
