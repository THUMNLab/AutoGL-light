import hyperopt

# from . import register_hpo
from .advisorbase import AdvisorBaseHPOptimizer
from .suggestion.algorithm.chocolate_grid_search import ChocolateGridSearchAlgorithm

# @register_hpo("randadvisor")


class GridAdvisorChoco(AdvisorBaseHPOptimizer):
    def __init__(self, args):
        super().__init__(args)
        self.method = ChocolateGridSearchAlgorithm()
