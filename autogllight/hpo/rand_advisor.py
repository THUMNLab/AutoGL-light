import hyperopt

from . import register_hpo
from .advisorbase import AdvisorBaseHPOptimizer
from .suggestion.algorithm.random_search import RandomSearchAlgorithm


@register_hpo("random")
class RandAdvisor(AdvisorBaseHPOptimizer):
    """
    Random search algorithm in `advisor` package
    See https://github.com/tobegit3hub/advisor for the package
    See .advisorbase.AdvisorBaseHPOptimizer for more information
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.method = RandomSearchAlgorithm()
