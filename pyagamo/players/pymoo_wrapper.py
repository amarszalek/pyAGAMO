import numpy as np
from pyagamo import Player
from pymoo.core.termination import NoTermination
from pymoo.core.problem import Problem
from pymoo.core.population import Population
from copy import deepcopy

class Algorithm_pymoo(Player):
    def __init__(self, num, npop, pymoo_alg, alg_kwargs={}, mq=0, random=True, ns=None, transport='ipc', verbose=False):
        self.pymoo_alg = pymoo_alg
        self.alg_kwargs = alg_kwargs
        super(Algorithm_pymoo, self).__init__(num, npop, mq=mq, random=random, ns=ns, transport=transport, verbose=verbose)
        
    def step(self, pop, pop_eval, pattern):
        evaluation_counter = 0
        s = np.sum(pattern)
        if s == 0:
            return pop, pop_eval, evaluation_counter
        indx = np.where(pattern)[0]
        xl = np.array([self.bounds[k][0] for k in indx])
        xu = np.array([self.bounds[k][1] for k in indx])
        self.problem = Problem(n_var=s, n_obj=1, n_constr=0, xl=xl, xu=xu)
        pymoo_pop = Population.empty()
        pymoo_pop = pymoo_pop.new("X", pop[:,pattern])
        pymoo_pop = pymoo_pop.set("F", pop_eval.reshape((-1,1)))
        alg = self.pymoo_alg(pop_size=pop.shape[0], sampling=pymoo_pop, **self.alg_kwargs)
        alg.setup(self.problem, termination=NoTermination())
        alg.tell(infills=pymoo_pop)
        pymoo_pop = alg.ask()
        new_pop = deepcopy(pop)
        X = pymoo_pop.get('X')
        inds = np.random.choice(X.shape[0], new_pop.shape[0])
        new_pop[:,pattern] = X[inds,:]
        if self.qclone is not None:
            new_pop, new_pop_old, new_pop_old_eval = self.qclone.get_new(new_pop)
        else:
            new_pop, new_pop_old, new_pop_old_eval = new_pop, None, None
        if new_pop.shape[0] > 0:
            new_pop = self.repair_call(new_pop)
            new_pop_eval = self.evaluate_call(new_pop)
            evaluation_counter += new_pop.shape[0]
            if self.qclone is not None:
                self.qclone.add_new(new_pop, new_pop_eval)
            if new_pop_old is not None:
                new_pop = np.concatenate((new_pop, new_pop_old))
                new_pop_eval = np.concatenate((new_pop_eval, new_pop_old_eval))
        elif new_pop_old is not None:
            new_pop = new_pop_old.copy()
            new_pop_eval = new_pop_old_eval.copy()
            
        return new_pop, new_pop_eval, evaluation_counter

   