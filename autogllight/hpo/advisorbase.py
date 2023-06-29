"""
HPO Module for tuning hyper parameters
"""

import json
import math
import time

from tqdm import trange

from .base import BaseHPOptimizer, TimeTooLimitedError
from .suggestion.algorithm.mocmaes import MocmaesAlgorithm
from .suggestion.algorithm.random_search import RandomSearchAlgorithm
from .suggestion.models import Study


class AdvisorBaseHPOptimizer(BaseHPOptimizer):
    """
    An abstract HPOptimizer using for `advisor` package.
    See https://github.com/tobegit3hub/advisor for more information

    Attributes
    ----------
    method : .suggestion.AbstractSuggestionAlgorithm
        The algorithm class in `suggestion`
    max_evals : int
        The max rounds of evaluating HPs

    Methods
    -------
    optimize
        Optimize the HP by the method within give model and HP space
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.method = None
        self.max_evals = kwargs.get("max_evals", 100)

    def _set_up(self, num_slaves, time_limit, memory_limit):
        """
        See .base.BaseHPOptimizer._set_up
        """
        self.new_trials = []
        self.xs = []
        self.best_id = None
        self.best_trainer = None
        space = self.hp_space
        current_config = self._encode_para(space)

        for i in range(num_slaves):
            self.new_trials.append(None)
            self.xs.append(None)

        study_configuration_json = {
            "goal": "MINIMIZE",
            "maxTrials": 5,
            "maxParallelTrials": 1,
            "randomInitTrials": 1,
            "params": current_config,
        }
        study_configuration = json.dumps(study_configuration_json)
        self.study = Study.create("HPO", study_configuration)
        self.config = study_configuration_json

    def _update_trials(self, pid, hp, perf):
        """
        See .base.BaseHPOptimizer._update_trials
        """
        self.new_trials[pid].status = "Completed"
        self.new_trials[pid].objective_value = perf
        if not self.best_id or perf < self.trials[self.best_id].objective_value:
            self.best_id = len(self.trials)
            self.best_hp = hp
        self.trials.append(self.new_trials[pid])

    def _get_suggestion(self, pid):
        """
        See .base.BaseHPOptimizer._get_suggestion
        """
        new_trials = self.method.get_new_suggestions(
            self.study, trials=self.trials, number=1
        )

        new_trial = new_trials[0]
        new_parameter_values_json = json.loads(new_trial.parameter_values)
        decoded_json, trial_para = self._decode_para(new_parameter_values_json)
        new_trial.parameter_values = json.dumps(trial_para)
        self.new_trials[pid] = new_trial
        self.xs[pid] = decoded_json
        return decoded_json

    def _best_hp(self):
        """
        See .base.BaseHPOptimizer._best_hp
        """
        if len(self.trials) == 0:
            return None, None
        best_perf = self.trials[self.best_id].objective_value
        decoded_json, _ = self._decode_para(
            json.loads(self.trials[self.best_id].parameter_values)
        )
        # self.logger.info("Best Parameter:")
        # self._print_info(decoded_json, best_perf)
        return self.best_trainer, best_perf

    def _setUp(self, current_config):
        study_configuration_json = {
            "goal": "MINIMIZE",
            "maxTrials": 5,
            "maxParallelTrials": 1,
            "randomInitTrials": 1,
            "params": current_config,
        }
        study_configuration = json.dumps(study_configuration_json)
        self.study = Study.create("HPO", study_configuration)
        self.config = study_configuration_json

    def optimize(self):
        """
        See .base.BaseHPOptimizer.optimize
        """
        self.trials = []
        if self.method == None:
            return

        space = self.hp_space
        current_space = self._encode_para(space)
        self._setUp(current_space)

        start_time = time.time()

        best_id = None
        best_trainer = None

        print("HPO Search Phase:\n")
        for i in trange(self.max_evals):
            if self.time_limit is not None and time.time() - start_time > self.time_limit:
                self.logger.info("Time out of limit, Epoch: {}".format(str(i)))
                break
            new_trials = self.method.get_new_suggestions(
                self.study, trials=self.trials, number=1
            )

            new_trial = new_trials[0]
            new_parameter_values_json = json.loads(new_trial.parameter_values)
            decoded_json, trial_para = self._decode_para(new_parameter_values_json)
            perf = self.f(decoded_json)
            new_trial.parameter_values = json.dumps(trial_para)
            new_trial.status = "Completed"
            new_trial.objective_value = perf
            if not best_id or perf < self.trials[best_id].objective_value:
                best_id = len(self.trials)
                best_hp = decoded_json
            else:
                del decoded_json
            self.trials.append(new_trial)
            # self._print_info(decoded_json, perf)

        if len(self.trials) == 0:
            raise TimeTooLimitedError(
                "Given time is too limited to finish one round in HPO."
            )

        best_perf = self.trials[best_id].objective_value
        # self.logger.info("Best Parameter:")

        return best_hp, best_perf
