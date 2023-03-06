from abc import abstractmethod
import torch.nn as nn
from nni.nas.pytorch import mutables
from nni.nas.pytorch.fixed import FixedArchitecture
import json
from copy import deepcopy
import torch

class OrderedMutable:
    """
    An abstract class with order, enabling to sort mutables with a certain rank.

    Parameters
    ----------
    order : int
        The order of the mutable
    """

    def __init__(self, order):
        self.order = order


class OrderedLayerChoice(OrderedMutable, mutables.LayerChoice):
    def __init__(
        self, order, op_candidates, reduction="sum", return_mask=False, key=None
    ):
        OrderedMutable.__init__(self, order)
        mutables.LayerChoice.__init__(
            self, op_candidates, reduction, return_mask, key)


class OrderedInputChoice(OrderedMutable, mutables.InputChoice):
    def __init__(
        self,
        order,
        n_candidates=None,
        choose_from=None,
        n_chosen=None,
        reduction="sum",
        return_mask=False,
        key=None,
    ):
        OrderedMutable.__init__(self, order)
        mutables.InputChoice.__init__(
            self, n_candidates, choose_from, n_chosen, reduction, return_mask, key
        )


class StrModule(nn.Module):
    """
    A shell used to wrap choices as nn.Module for non-one-shot space definition
    You can use ``map_nn`` function

    Parameters
    ----------
    name : anything
        the name of module, can be any type
    """

    def __init__(self, name):
        super().__init__()
        self.str = name

    def forward(self, *args, **kwargs):
        return self.str

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.str)


def map_nn(names):
    """
    A function used to wrap choices as nn.Module for non-one-shot space definition

    Parameters
    ----------
    name : list of anything
        the names of module, can be any type
    """
    return [StrModule(x) for x in names]


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

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Define the forward pass of space model
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

    def setLayerChoice(
        self, order, op_candidates, reduction="sum", return_mask=False, key=None
    ):
        """
        Give a unique key if not given
        """
        orikey = key
        if orikey == None:
            key = f"default_key_{self._default_key}"
            self._default_key += 1
            orikey = key
        layer = OrderedLayerChoice(
            order, op_candidates, reduction, return_mask, orikey)
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
        orikey = key
        if orikey == None:
            key = f"default_key_{self._default_key}"
            self._default_key += 1
            orikey = key
        layer = OrderedInputChoice(
            order, n_candidates, choose_from, n_chosen, reduction, return_mask, orikey
        )
        return layer

class FixedInputChoice(nn.Module):
    """
    Use to replace `InputChoice` Mutable in fix process

    Parameters
    ----------
    mask : list
        The mask indicating which input to choose
    """

    def __init__(self, mask):
        self.mask_len = len(mask)
        for i in range(self.mask_len):
            if mask[i]:
                self.selected = i
                break
        super().__init__()

    def forward(self, optional_inputs):
        if len(optional_inputs) == self.mask_len:
            return optional_inputs[self.selected]


class CleanFixedArchitecture(FixedArchitecture):
    """
    Fixed architecture mutator that always selects a certain graph, allowing deepcopy

    Parameters
    ----------
    model : nn.Module
        A mutable network.
    fixed_arc : dict
        Preloaded architecture object.
    strict : bool
        Force everything that appears in ``fixed_arc`` to be used at least once.
    verbose : bool
        Print log messages if set to True
    """

    def __init__(self, model, fixed_arc, strict=True, verbose=True):
        super().__init__(model, fixed_arc, strict, verbose)

    def replace_all_choice(self, module=None, prefix=""):
        """
        Replace all choices with selected candidates. It's done with best effort.
        In case of weighted choices or multiple choices. if some of the choices on weighted with zero, delete them.
        If single choice, replace the module with a normal module.

        Parameters
        ----------
        module : nn.Module
            Module to be processed.
        prefix : str
            Module name under global namespace.
        """
        
        if module is None:
            module = self.model
        for name, mutable in module.named_children():
            global_name = (prefix + "." if prefix else "") + name
            if isinstance(mutable, OrderedLayerChoice):
                chosen = self._fixed_arc[mutable.key]
                if sum(chosen) == 1 and max(chosen) == 1 and not mutable.return_mask:
                    # sum is one, max is one, there has to be an only one
                    # this is compatible with both integer arrays, boolean arrays and float arrays
                    setattr(module, name, mutable[chosen.index(1)])
                else:
                    # remove unused parameters
                    for ch, n in zip(chosen, mutable.names):
                        if ch == 0 and not isinstance(ch, float):
                            setattr(mutable, n, None)
            elif isinstance(mutable, OrderedInputChoice):
                chosen = self._fixed_arc[mutable.key]
                setattr(module, name, FixedInputChoice(chosen))
            else:
                self.replace_all_choice(mutable, global_name)


def apply_fixed_architecture(model, fixed_arc, verbose=True):
    """
    Load architecture from `fixed_arc` and apply to model.

    Parameters
    ----------
    model : torch.nn.Module
        Model with mutables.
    fixed_arc : str or dict
        Path to the JSON that stores the architecture, or dict that stores the exported architecture.
    verbose : bool
        Print log messages if set to True

    Returns
    -------
    FixedArchitecture
        Mutator that is responsible for fixes the graph.
    """

    if isinstance(fixed_arc, str):
        with open(fixed_arc) as f:
            fixed_arc = json.load(f)
    architecture = CleanFixedArchitecture(model, fixed_arc, verbose)
    architecture.reset()

    # for the convenience of parameters counting
    architecture.replace_all_choice()
    return architecture
