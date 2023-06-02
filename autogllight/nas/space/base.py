from abc import abstractmethod
import torch.nn as nn
import json
from copy import deepcopy
import torch
from .nni import (
    apply_fixed_architecture,
    OrderedLayerChoice,
    OrderedInputChoice,
)


class BoxModel(nn.Module):
    """
    The box wrapping a space, can be passed to later procedure or trainer

    Parameters
    ----------
    space_model : BaseSpace
        The space which should be wrapped
    device : str or torch.device
        The device to place the model
    """

    def __init__(self, space_model, *args, **kwargs):
        super().__init__()
        self.init = True
        self.space = []
        self.hyperparams = {}
        self._model = space_model
        self.num_features = self._model.input_dim
        self.num_classes = self._model.output_dim
        self.params = {"num_class": self.num_classes, "features_num": self.num_features}
        self.selection = None

    def fix(self, selection):
        """
        To fix self._model with a selection

        Parameters
        ----------
        selection : dict
            A seletion indicating the choices of mutables
        """
        self.selection = selection
        self._model.instantiate()
        apply_fixed_architecture(self._model, selection, verbose=False)
        return self

    def forward(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    def __repr__(self) -> str:
        return str({"model": self.model, "selection": self.selection})


class BaseSpace(nn.Module):
    """
    Base space class of NAS module. Defining space containing all models.
    Please use mutables to define your whole space. Refer to
    `https://nni.readthedocs.io/en/stable/NAS/WriteSearchSpace.html`
    for detailed information.

    Parameters
    ----------
    init: bool
        Whether to initialize the whole space. Default: `False`
    """

    def __init__(self):
        super().__init__()
        self._initialized = False
        self._default_key = 0

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Define the forward pass of space model
        """
        raise NotImplementedError()

    @abstractmethod
    def instantiate(self):
        """
        Instantiate modules in the space
        """
        raise NotImplementedError()

    def getOriKey(self, key):
        orikey = key
        if orikey == None:
            key = f"default_key_{self._default_key}"
            self._default_key += 1
            orikey = key
        return orikey

    def setLayerChoice(
        self, order, op_candidates, reduction="sum", return_mask=False, key=None
    ):
        """
        Give a unique key if not given
        """
        key = self.getOriKey(key)
        layer = OrderedLayerChoice(order, op_candidates, reduction, return_mask, key)
        setattr(self, key, layer)
        return layer

    def setInputChoice(
        self,
        order,
        n_candidates=None,
        choose_from=None,
        n_chosen=None,
        reduction="sum",
        return_mask=False,
        key=None,
    ):
        """
        Give a unique key if not given
        """
        key = self.getOriKey(key)
        layer = OrderedInputChoice(
            order, n_candidates, choose_from, n_chosen, reduction, return_mask, key
        )
        setattr(self, key, layer)
        return layer

    def parse_model(self, selection):
        """Get the fixed model from the selection
        Usage: the fixed model can be obtained by boxmodel._model 
        Warning : this method will randomize the learnable parameters in the model, as the model is re-instantiated.
        """
        boxmodel = BoxModel(self).fix(selection)
        return boxmodel


"""
BoxModel is the space itself, but without replacing the operation choices.
Therefore, the choices in BoxModel is subclasses of Mutables, which can be collected in functions like apply_fixed_architecture.
In this way, the fixed architecture "BoxModel" will get fixed operations, while the original space will not be changed.
Moreover, the fixed architecture "BoxModel" will not have multiple operations mixed together as in DARTS.  
"""
