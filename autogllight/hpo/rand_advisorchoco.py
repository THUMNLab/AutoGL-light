import hyperopt

from . import register_hpo
from .advisorbase import AdvisorBaseHPOptimizer
from .suggestion.algorithm.chocolate_random_search import ChocolateRandomSearchAlgorithm

# @register_hpo("randomchoco")


class RandAdvisorChoco(AdvisorBaseHPOptimizer):
    def __init__(self, args):
        super().__init__(args)
        self.method = ChocolateRandomSearchAlgorithm()
