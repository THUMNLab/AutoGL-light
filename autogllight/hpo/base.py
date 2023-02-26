"""
HPO Module for tuning hyper parameters
"""

import random

class HPSpace:
    def __init__(self, name):
        self.name = name

class RangeHP(HPSpace):
    def __init__(self, name, min_value, max_value):
        super.__init__(name)
        self.min = min_value
        self.max = max_value

class LogRangeHP(HPSpace):
    def __init__(self, name, min_value, max_value):
        super.__init__(name)
        self.min = min_value
        self.max = max_value

class ChoiceHP(HPSpace):
    def __init__(self, name, choices):
        super.__init__(name)
        self.choices = choices

class BaseHPOptimizer:
    def __init__(self, hp_space, f):
        self.hp_space = hp_space
        self.f = f
        self.trails = []

    def optimize(self):
        """Key function. Return the best hp & performance"""
        return None         

    def all_trails(self):
        return self.trails

def test():
    def f(lr, opti):
        return lr
    space = [
        LogRangeHP("lr", 0.001, 0.1),
        ChoiceHP("opti", ['adam', 'sgd'])
    ]
    bhpo = BaseHPOptimizer(space, f)
    bhpo.optimize()

if __name__ == "__main__":
    test()
     
