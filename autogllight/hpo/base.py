"""
HPO Module for tuning hyper parameters
"""

import math
import multiprocessing as mp
import random
import time
from abc import ABC, abstractmethod

from .suggestion.models import Trial


class HPSpace(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def to_config(self):
        pass


class RangeHP(HPSpace):
    def __init__(self, name, min_value, max_value):
        super.__init__(name)
        self.min = min_value
        self.max = max_value

    def to_config(self):
        return {
            "parameterName": self.name,
            "type": "DOUBLE",
            "minValue": self.min,
            "maxValue": self.max,
            "scalingType": "LINEAR",
        }


class LogRangeHP(HPSpace):
    def __init__(self, name, min_value, max_value):
        super().__init__(name)
        self.min = min_value
        self.max = max_value

    def to_config(self):
        return {
            "parameterName": self.name,
            "type": "DOUBLE",
            "minValue": self.min,
            "maxValue": self.max,
            "scalingType": "LOG",
        }


class ChoiceHP(HPSpace):
    def __init__(self, name, choices):
        super().__init__(name)
        self.choices = choices

    def to_config(self):
        return {
            "parameterName": self.name,
            "type": "CATEGORICAL",
            "feasiblePoints": self.choices,
        }


class BaseHPOptimizer:
    def __init__(self, hp_space, f, time_limit=None, memory_limit=None, max_evals=None):
        self.hp_space = self._hpspace_to_config(hp_space)
        self.f = f
        self.trials = []
        self.time_limit = time_limit
        self.memory_limit = memory_limit
        self.max_evals = max_evals

    def _hpspace_to_config(self, hp_space):
        tmp = []
        for hp in hp_space:
            tmp.append(hp.to_config())
        return {"space": tmp}

    def _decompose_dld(self, config):
        self._dld = {}
        if isinstance(config, list):
            return config
        # config is dict
        list_config = []
        for key in config:
            self._dld[key] = []
            for para in config[key]:
                self._dld[key].append(para["parameterName"])
                newpara = para.copy()
                newpara["parameterName"] = key + ":" + para["parameterName"]
                if "cutPara" in para.keys():
                    if isinstance(newpara["cutPara"], str):
                        newpara["cutPara"] = key + ":" + para["cutPara"]
                    else:
                        newpara["cutPara"] = [
                            key + ":" + cutname for cutname in para["cutPara"]
                        ]
                list_config.append(newpara)
        return list_config

    def _compose_dld(self, para):
        if not self._dld:
            return para
        fin = {}
        for key in self._dld:
            fin[key] = {}
            for pname in self._dld[key]:
                fin[key][pname] = para[key + ":" + pname]
        return fin

    def _decompose_depend_list_para(self, config):
        self._depend_map = {}

        def get_depended_para(name):
            for p in config:
                if p["parameterName"] == name:
                    return
            raise WrongDependedParameterError("The depended parameter does not exist.")

        for para in config:
            if para["type"] in ("NUMERICAL_LIST", "CATEGORICAL_LIST") and para.get(
                "cutPara", None
            ):
                self._depend_map[para["parameterName"]] = para
                if type(para["cutPara"]) == str:
                    get_depended_para(para["cutPara"])
                else:
                    for dpara in para["cutPara"]:
                        get_depended_para(dpara)

        return config

    def _compose_depend_list_para(self, config):
        for para in self._depend_map:
            cutparas = self._depend_map[para]["cutPara"]
            if type(cutparas) == str:
                dparas = [config[cutparas]]
                # dparas = config[cutparas]
            else:
                dparas = []
                for dpara in cutparas:
                    dparas.append(config[dpara])
            paralen = self._depend_map[para]["cutFunc"](dparas)
            config[para] = config[para][:paralen]
        return config

    def _decompose_list_fixed_para(self, config):
        fin = []
        self._list_map = {}
        self._fix_map = {}
        for para in config:
            if para["type"] == "NUMERICAL_LIST":
                self._list_map[para["parameterName"]] = para["length"]
                if type(para["minValue"]) != list:
                    para["minValue"] = [para["minValue"] for i in range(para["length"])]
                if type(para["maxValue"]) != list:
                    para["maxValue"] = [para["maxValue"] for i in range(para["length"])]
                for i, x, y in zip(
                    range(para["length"]), para["minValue"], para["maxValue"]
                ):
                    new_para = {}
                    new_para["parameterName"] = para["parameterName"] + "_" + str(i)
                    new_para["type"] = para["numericalType"]
                    new_para["minValue"] = x
                    new_para["maxValue"] = y
                    new_para["scalingType"] = para["scalingType"]
                    fin.append(new_para)
            elif para["type"] == "CATEGORICAL_LIST":
                self._list_map[para["parameterName"]] = para["length"]
                category = para["feasiblePoints"]
                self._category_map[para["parameterName"]] = category

                cur_points = ",".join(map(lambda _x: str(_x), range(len(category))))
                for i in range(para["length"]):
                    new_para = dict()
                    new_para["parameterName"] = para["parameterName"] + "_" + str(i)
                    new_para["type"] = "DISCRETE"
                    new_para["feasiblePoints"] = cur_points
                    fin.append(new_para)
            elif para["type"] == "FIXED":
                self._fix_map[para["parameterName"]] = para["value"]
            else:
                new_para = para.copy()
                new_para["parameterName"] = para["parameterName"] + "_"
                fin.append(new_para)
        return fin

    def _compose_list_fixed_para(self, config):
        fin = {}
        # compose list
        for pname in self._list_map:
            val = []
            for i in range(self._list_map[pname]):
                val.append(config[pname + "_" + str(i)])
                del config[pname + "_" + str(i)]
            if pname in self._category_map:
                val = [self._category_map[pname][i] for i in val]
            fin[pname] = val
        # deal other para
        for pname in config:
            fin[pname[:-1]] = config[pname]
        for pname in self._fix_map:
            fin[pname] = self._fix_map[pname]
        return fin

    def _encode_para_convert(self, config):
        """
        Convert all types of para space to DOUBLE(linear), DISCRETE
        config: [{
            "parameterName": "num_layers",
            "type": "DISCRETE",
            "feasiblePoints": "1,2,3,4",
        },{
            "parameterName": "hidden",
            "type": "NUMERICAL_LIST",
            "numericalType": "INTEGER",
            "length": 4,
            "minValue": [4, 4, 4, 4],
            "maxValue": [32, 32, 32, 32],
            "scalingType": "LOG"
        },{
            "parameterName": "dropout",
            "type": "DOUBLE",
            "minValue": 0.1,
            "maxValue": 0.9,
            "scalingType": "LINEAR"
        }]"""
        self._category_map = {}
        self._discrete_map = {}
        self._numerical_map = {}

        current_config = []
        for para in config:
            if para["type"] == "DOUBLE" or para["type"] == "INTEGER":
                cur_para = para.copy()
                cur_para["type"] = "DOUBLE"
                if para["scalingType"] == "LOG":
                    cur_para["minValue"] = math.log(para["minValue"])
                    cur_para["maxValue"] = math.log(para["maxValue"])
                current_config.append(cur_para)
                self._numerical_map[para["parameterName"]] = para
            elif para["type"] == "CATEGORICAL" or para["type"] == "DISCRETE":
                if para["type"] == "DISCRETE":
                    cate_list = para["feasiblePoints"].split(",")
                    cate_list = list(map(lambda x: x.strip(), cate_list))
                else:
                    cate_list = para["feasiblePoints"]
                cur_points = ",".join(map(lambda x: str(x), range(len(cate_list))))
                cur_para = para.copy()
                cur_para["feasiblePoints"] = cur_points
                cur_para["type"] = "DISCRETE"
                current_config.append(cur_para)
                if para["type"] == "CATEGORICAL":
                    self._category_map[para["parameterName"]] = cate_list
                else:
                    self._discrete_map[para["parameterName"]] = cate_list
            else:
                current_config.append(para)
        return current_config

    def _decode_para_convert(self, para):
        """
        decode HPO given para to user(externel) para and trial para
        """
        externel_para = para.copy()
        trial_para = para.copy()
        for name in para:
            if name in self._numerical_map:
                old_para = self._numerical_map[name]
                val = para[name]
                if old_para["scalingType"] == "LOG":
                    val = math.exp(val)
                    if val < old_para["minValue"]:
                        val = old_para["minValue"]
                    elif val > old_para["maxValue"]:
                        val = old_para["maxValue"]
                if old_para["type"] == "INTEGER":
                    val = int(round(val))
                externel_para[name] = val
                trial_para[name] = (
                    val if old_para["scalingType"] != "LOG" else math.log(val)
                )
            elif name in self._category_map:
                externel_para[name] = self._category_map[name][int(para[name])]
                trial_para[name] = para[name]
            elif name in self._discrete_map:
                externel_para[name] = eval(self._discrete_map[name][int(para[name])])
                trial_para[name] = para[name]
        return externel_para, trial_para

    def _encode_para(self, config):
        # dict(list(dict)) -> list(dict)
        config = self._decompose_dld(config)
        # (dependent list) -> list
        config = self._decompose_depend_list_para(config)
        # list -> double/discrete; fixed -> removed
        config = self._decompose_list_fixed_para(config)
        # discrete -> categorical; log -> linear
        config = self._encode_para_convert(config)
        return config

    def _decode_para(self, para):
        para, trial_para = self._decode_para_convert(para)
        para = self._compose_list_fixed_para(para)
        para = self._compose_depend_list_para(para)
        para = self._compose_dld(para)
        return para, trial_para

    def slave(self, pipe):
        while 1:
            x = pipe.recv()
            y = self.f(x)
            pipe.send((x, y))

    def _set_up(self, num_slaves, time_limit, memory_limit):
        """
        Initialize something used in "optimize"

        Parameters
        ----------
        trainer : ..train.BaseTrainer
            Including model, giving HP space and using for training
        dataset : ...datasets
            Dataset to train and evaluate.
        time_limit : int
            Max time to run HPO
        memory_limit : None
            No implementation yet
        """
        pass

    def _update_trials(self, pid, hp, perf):
        """
        After the evaluation phase of each turn, update history trials according to the performance

        Parameters
        ----------
        pid : int
            The process id which runs the evaluation
        hp : dict
            The HPs used in evaluation
        perf : float
            The performance of the HP, higher is better
        """
        pass

    def _get_suggestion(self, pid):
        """
        Give the next HP suggestion

        Parameters
        ----------
        pid : int
            The process id which will run the evaluation

        Returns
        -------
        para_json: dict
            The suggested HP
        """
        pass

    def _best_hp(self):
        """
        Give the best HP and the best trainer as the returns of "optimize"

        Returns
        -------
        trainer: ..train.BaseTrainer
            The trainer including the best trained model
        para_json: dict
            The best HP
        """
        pass

    def optimize(self):
        """Key function. Return the best hp & performance"""
        start_time = time.time()
        cur_evals = 0

        slaves = 2
        pipes = []
        procs = []
        statuses = []
        for i in range(slaves):
            ps, pr = mp.Pipe()
            pipes.append(ps)
            proc = mp.Process(
                target=self.slave,
                args=(pr,),
            )
            proc.start()
            procs.append(proc)
            statuses.append("ready")

        # do something of the certain HPO algo
        self._set_up(slaves, self.time_limit, self.memory_limit)

        while cur_evals < self.max_evals:
            # timeout
            if time.time() - start_time > self.time_limit:
                self.logger.info("Time out of limit, Epoch: {}".format(str(cur_evals)))
                break
            for i in range(slaves):
                sent = pipes[i].poll()
                if sent or statuses[i] == "ready":
                    if sent:
                        x, y = pipes[i].recv()
                        self.trials.append(
                            self._creat_a_trail("HPO", "Base", "Completed", x, y)
                        )
                        self._update_trials(i, x, y)
                        cur_evals += 1
                        if cur_evals >= self.max_evals:
                            break
                    else:
                        statuses[i] = "running"
                    x = self._get_suggestion(i)
                    pipes[i].send(x)

        for i in range(slaves):
            procs[i].terminate()

        best_x, best_perf = self._best_hp()
        if best_x == None:
            raise TimeTooLimitedError(
                "Given time is too limited to finish one round in HPO."
            )
        return best_x, best_perf

    def _creat_a_trail(
        self, study_name, name, status, parametrer_values, objective_value
    ):
        trial = Trial.create(study_name, name)
        trial.status = status
        trial.parameter_values = parametrer_values
        trial.objective_value = objective_value
        return trial

    def all_trials(self):
        raise self.trials


class TimeTooLimitedError(Exception):
    pass


class WrongDependedParameterError(Exception):
    pass


def f(hps: dict, dataset=None):
    # build a module with hps
    # if dataset is not None, use the provided dataset, or use the dataset from trainer otherwise
    model = get_model(hps["num_layers"])
    trainer = Trainer(data)
    trainer.set_hp(hps)
    trainer.train(model)
    return trainer.evaluate(model)


def test():

    space = [
        LogRangeHP("lr", 0.001, 0.1),
        ChoiceHP("opti", ["adam", "sgd"]),
        ChoiceHP("num_layers", [2, 3]),
    ]
    bhpo = BaseHPOptimizer(space, f)
    bhpo.optimize()


if __name__ == "__main__":
    test()
