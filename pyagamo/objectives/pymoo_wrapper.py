import numpy as np
from pyagamo import Objective

class Problem_pymoo(Objective):
    def __init__(self, num, obj, pymoo_prob, ns=None, transport='ipc', args=None, verbose=False):
        self.obj = obj - 1
        self.n_var = pymoo_prob.n_var
        self.n_obj = pymoo_prob.n_obj
        self.prob = pymoo_prob
        self.bounds = list(zip(self.prob.xl, self.prob.xu))
        super(Problem_pymoo, self).__init__(num, ns, transport, args, verbose)

    def call(self, x, *args):
        res = self.prob.evaluate(x)
        return res[:, self.obj]