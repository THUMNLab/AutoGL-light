"""
This file includes modified mutables for supporting ordered operations and also advanced choices. 
"""

from . import mutables
from .fixed import FixedArchitecture
import json
import logging
from torch import nn
from .mutables import Mutable, InputChoice, LayerChoice

_logger = logging.getLogger(__name__)


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
    def __init__(self, order, op_candidates, reduction="sum", return_mask=False, key=None):
        OrderedMutable.__init__(self, order)
        mutables.LayerChoice.__init__(self, op_candidates, reduction, return_mask, key)


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
        mutables.InputChoice.__init__(self, n_candidates, choose_from, n_chosen, reduction, return_mask, key)


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
        assert self.sampled is not None, "At least one path needs to be sampled before fprop."
        if isinstance(self.sampled, list):
            return sum([getattr(self, self.op_names[i])(*args, **kwargs) for i in self.sampled])  # pylint: disable=not-an-iterable
        else:
            return getattr(self, self.op_names[self.sampled])(*args, **kwargs)  # pylint: disable=invalid-sequence-index

    def sampled_choices(self):
        if self.sampled is None:
            return []
        elif isinstance(self.sampled, list):
            return [getattr(self, self.op_names[i]) for i in self.sampled]  # pylint: disable=not-an-iterable
        else:
            return [getattr(self, self.op_names[self.sampled])]  # pylint: disable=invalid-sequence-index

    def __len__(self):
        return len(self.op_names)

    @property
    def mask(self):
        return _get_mask(self.sampled, len(self))

    def __repr__(self):
        return f"PathSamplingLayerChoice(op_names={self.op_names}, chosen={self.sampled})"


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
            return sum([input_tensors[t] for t in self.sampled])  # pylint: disable=not-an-iterable
        else:
            return input_tensors[self.sampled]

    def __len__(self):
        return self.n_candidates

    @property
    def mask(self):
        return _get_mask(self.sampled, len(self))

    def __repr__(self):
        return f"PathSamplingInputChoice(n_candidates={self.n_candidates}, chosen={self.sampled})"


def get_module_order(root_module):
    key2order = {}

    def apply(m):
        for name, child in m.named_children():
            if isinstance(child, Mutable):
                key2order[child.key] = child.order
            else:
                apply(child)

    apply(root_module)
    return key2order


def sort_replaced_module(k2o, modules):
    modules = sorted(modules, key=lambda x: k2o[x[0]])
    return modules


def _replace_module_with_type(root_module, init_fn, type_name, modules):
    if modules is None:
        modules = []

    def apply(m):
        for name, child in m.named_children():
            if isinstance(child, type_name):
                setattr(m, name, init_fn(child))
                modules.append((child.key, getattr(m, name)))
            else:
                apply(child)

    apply(root_module)
    return modules


def replace_layer_choice(root_module, init_fn, modules=None):
    """
    Replace layer choice modules with modules that are initiated with init_fn.

    Parameters
    ----------
    root_module : nn.Module
        Root module to traverse.
    init_fn : Callable
        Initializing function.
    modules : dict, optional
        Update the replaced modules into the dict and check duplicate if provided.

    Returns
    -------
    List[Tuple[str, nn.Module]]
        A list from layer choice keys (names) and replaced modules.
    """
    return _replace_module_with_type(root_module, init_fn, (LayerChoice, ), modules)


def replace_input_choice(root_module, init_fn, modules=None):
    """
    Replace input choice modules with modules that are initiated with init_fn.

    Parameters
    ----------
    root_module : nn.Module
        Root module to traverse.
    init_fn : Callable
        Initializing function.
    modules : dict, optional
        Update the replaced modules into the dict and check duplicate if provided.

    Returns
    -------
    List[Tuple[str, nn.Module]]
        A list from layer choice keys (names) and replaced modules.
    """
    return _replace_module_with_type(root_module, init_fn, (InputChoice, ), modules)
