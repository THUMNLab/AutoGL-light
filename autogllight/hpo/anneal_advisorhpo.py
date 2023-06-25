import hyperopt

from . import register_hpo
from .advisorbase import AdvisorBaseHPOptimizer
from .suggestion.algorithm.simulate_anneal import SimulateAnnealAlgorithm


@register_hpo("anneal")
class AnnealAdvisorHPO(AdvisorBaseHPOptimizer):
    """
    Simulate anneal algorithm in `advisor` package
    See https://github.com/tobegit3hub/advisor for the package
    See .advisorbase.AdvisorBaseHPOptimizer for more information
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.method = SimulateAnnealAlgorithm()
