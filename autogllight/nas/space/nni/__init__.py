from .choice_ordered import (
    apply_fixed_architecture,
    OrderedMutable,
    OrderedLayerChoice,
    OrderedInputChoice,
    get_module_order,
    sort_replaced_module,
    replace_layer_choice,
    replace_input_choice,
)

from .choice_path import PathSamplingLayerChoice, PathSamplingInputChoice
from .choice_darts import DartsLayerChoice, DartsInputChoice
