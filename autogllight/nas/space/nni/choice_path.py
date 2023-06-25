from torch import nn
import torch


class PathSamplingLayerChoice(nn.Module):
    """
    Mixed module, in which fprop is decided by exactly one or multiple (sampled) module.
    If multiple module is selected, the result will be sumed and returned.

    Attributes
    ----------
    sampled : int or list of int
        Sampled module indices.
    mask : tensor
        A multi-hot bool 1D-tensor representing the sampled mask.
    """

    def __init__(self, layer_choice):
        super(PathSamplingLayerChoice, self).__init__()
        self.op_names = []
        for name, module in layer_choice.named_children():
            self.add_module(name, module)
            self.op_names.append(name)
        assert self.op_names, "There has to be at least one op to choose from."
        self.sampled = None  # sampled can be either a list of indices or an index

    def forward(self, *args, **kwargs):
        assert (
            self.sampled is not None
        ), "At least one path needs to be sampled before fprop."
        if isinstance(self.sampled, list):
            return sum(
                [getattr(self, self.op_names[i])(*args, **kwargs) for i in self.sampled]
            )  # pylint: disable=not-an-iterable
        else:
            return getattr(self, self.op_names[self.sampled])(
                *args, **kwargs
            )  # pylint: disable=invalid-sequence-index

    def sampled_choices(self):
        if self.sampled is None:
            return []
        elif isinstance(self.sampled, list):
            return [
                getattr(self, self.op_names[i]) for i in self.sampled
            ]  # pylint: disable=not-an-iterable
        else:
            return [
                getattr(self, self.op_names[self.sampled])
            ]  # pylint: disable=invalid-sequence-index

    def __len__(self):
        return len(self.op_names)

    @property
    def mask(self):
        return _get_mask(self.sampled, len(self))

    def __repr__(self):
        return (
            f"PathSamplingLayerChoice(op_names={self.op_names}, chosen={self.sampled})"
        )


class PathSamplingInputChoice(nn.Module):
    """
    Mixed input. Take a list of tensor as input, select some of them and return the sum.

    Attributes
    ----------
    sampled : int or list of int
        Sampled module indices.
    mask : tensor
        A multi-hot bool 1D-tensor representing the sampled mask.
    """

    def __init__(self, input_choice):
        super(PathSamplingInputChoice, self).__init__()
        self.n_candidates = input_choice.n_candidates
        self.n_chosen = input_choice.n_chosen
        self.sampled = None

    def forward(self, input_tensors):
        if isinstance(self.sampled, list):
            return sum(
                [input_tensors[t] for t in self.sampled]
            )  # pylint: disable=not-an-iterable
        else:
            return input_tensors[self.sampled]

    def __len__(self):
        return self.n_candidates

    @property
    def mask(self):
        return _get_mask(self.sampled, len(self))

    def __repr__(self):
        return f"PathSamplingInputChoice(n_candidates={self.n_candidates}, chosen={self.sampled})"
