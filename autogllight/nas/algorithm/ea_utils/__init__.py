import dataclasses
import random
import numpy as np

@dataclasses.dataclass
class Individual:
    """
    A class that represents an individual.
    Holds two attributes, where ``x`` is the model and ``y`` is the metric (e.g., accuracy).
    """
    x: dict
    y: float


class MutationSampler:
    """uniform mutator

    Parameters
    ----------
    nas_modules: 
        nas_modules in NAS algorithms , including choices of modules
    mutation_prob: float
        probability of doing mutation in each choice.
    parent : dict
        parent individual's choices
    """
    def __init__(self,nas_modules,mutation_prob):
        selection_range = {}
        for k, v in nas_modules:
            selection_range[k] = len(v)
        self.selection_dict = selection_range
        self.mutation_prob = mutation_prob

    def resample(self, parent):
        search_space=self.selection_dict
        child = {}
        for k, v in parent.items():
            if random.uniform(0, 1) < self.mutation_prob:
                child[k] = np.random.choice(range(search_space[k])) # do not exclude the original operator
            else:
                child[k] = v
        return child

class UniformSampler:
    """Uniform Sampler

    Parameters
    ----------
    nas_modules: 
        nas_modules in NAS algorithms , including choices of modules
    """
    def __init__(self,nas_modules):
        selection_range = {}
        for k, v in nas_modules:
            selection_range[k] = len(v)
        self.selection_dict = selection_range
    def resample(self):
        selection = {}
        for k, v in self.selection_dict.items():
            selection[k] = np.random.choice(range(v))
        return selection