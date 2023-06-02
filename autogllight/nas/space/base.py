from abc import abstractmethod
import torch.nn as nn
import json
from copy import deepcopy
import torch
from .nni import (
    apply_fixed_architecture, 
    OrderedMutable, 
    OrderedLayerChoice, 
    OrderedInputChoice,
    get_module_order,
    sort_replaced_module,
    PathSamplingLayerChoice,
    PathSamplingInputChoice,
    replace_layer_choice,
    replace_input_choice
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

    def __init__(self, space_model, device):
        super().__init__()
        self.init = True
        self.space = []
        self.hyperparams = {}
        self._model = space_model
        self.num_features = self._model.input_dim
        self.num_classes = self._model.output_dim
        self.params = {"num_class": self.num_classes, "features_num": self.num_features}
        self.selection = None

    def _initialize(self):
        return True

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

    def from_hyper_parameter(self, hp):
        """
        receive no hp, just copy self and reset the learnable parameters.
        """

        ret_self = deepcopy(self)
        ret_self._model.instantiate()
        if ret_self.selection:
            apply_fixed_architecture(ret_self._model, ret_self.selection, verbose=False)
        return ret_self

    def __repr__(self) -> str:
        return str(
            {'parameter': get_hardware_aware_metric(self.model, 'parameter'),
             'model': self.model,
             'selection': self.selection
             })
        
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

    def fix(self, selection):
        """
        To fix self._model with a selection

        Parameters
        ----------
        selection : dict
            A seletion indicating the choices of mutables
        """
        self.selection = selection
        apply_fixed_architecture(self, selection, verbose=False)
        return self

    def getOriKey(self, key):
        orikey = key
        if orikey == None:
            key = f"default_key_{self._default_key}"
            self._default_key += 1
            orikey = key
        return orikey

    def setLayerChoice(self, order, op_candidates, reduction="sum", return_mask=False, key=None):
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
        layer = OrderedInputChoice(order, n_candidates, choose_from, n_chosen, reduction, return_mask, key)
        setattr(self, key, layer)
        return layer
    
    def wrap(self):
        """
        Return a BoxModel which wrap self as a model
        Used to pass to trainer
        To use this function, must contain `input_dim` and `output_dim`
        """
        device = next(self.parameters()).device
        return BoxModel(self, device)
    

