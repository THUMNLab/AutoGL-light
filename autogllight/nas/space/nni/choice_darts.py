from torch import nn
import torch
from torch.nn import functional as F


class DartsLayerChoice(nn.Module):
    def __init__(self, layer_choice):
        super(DartsLayerChoice, self).__init__()
        self.name = layer_choice.key
        self.op_choices = nn.ModuleDict(layer_choice.named_children())
        self.alpha = nn.Parameter(torch.randn(len(self.op_choices)) * 1e-3)

    def forward(self, *args, **kwargs):
        op_results = torch.stack(
            [op(*args, **kwargs) for op in self.op_choices.values()]
        )
        alpha_shape = [-1] + [1] * (len(op_results.size()) - 1)
        return torch.sum(op_results * F.softmax(self.alpha, -1).view(*alpha_shape), 0)

    def parameters(self):
        for _, p in self.named_parameters():
            yield p

    def named_parameters(self):
        for name, p in super(DartsLayerChoice, self).named_parameters():
            if name == "alpha":
                continue
            yield name, p

    def export(self):
        return torch.argmax(self.alpha).item()


class DartsInputChoice(nn.Module):
    def __init__(self, input_choice):
        super(DartsInputChoice, self).__init__()
        self.name = input_choice.key
        self.alpha = nn.Parameter(torch.randn(input_choice.n_candidates) * 1e-3)
        self.n_chosen = input_choice.n_chosen or 1

    def forward(self, inputs):
        inputs = torch.stack(inputs)
        alpha_shape = [-1] + [1] * (len(inputs.size()) - 1)
        return torch.sum(inputs * F.softmax(self.alpha, -1).view(*alpha_shape), 0)

    def parameters(self):
        for _, p in self.named_parameters():
            yield p

    def named_parameters(self):
        for name, p in super(DartsInputChoice, self).named_parameters():
            if name == "alpha":
                continue
            yield name, p

    def export(self):
        return torch.argsort(-self.alpha).cpu().numpy().tolist()[: self.n_chosen]
