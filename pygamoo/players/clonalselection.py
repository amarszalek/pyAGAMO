import numpy as np
from pygamoo import Player
from copy import deepcopy


class ClonalSelection(Player):
    def __init__(self, num, obj_queue, repair_queue, cmd_exchange, npop, nvars, bounds, host, port, player_parm):
        self.nclone = player_parm['nclone']
        self.mutate_args = tuple(player_parm['mutate_args'])
        super(ClonalSelection, self).__init__(num, obj_queue, repair_queue, cmd_exchange, npop, nvars, bounds, host, port)

    def step(self, pop, pop_eval, pattern):
        temp_pop = deepcopy(pop)
        temp_pop_eval = deepcopy(pop_eval)
        arg_sort = temp_pop_eval.argsort()
        indices = []
        better = []
        better_eval = []
        evaluation_counter = 0
        for rank, arg in enumerate(arg_sort):
            clone_num = max(int(self.nclone / (rank + 1) + 0.5), 1)
            clones = np.array([self._mutate(temp_pop[arg], pattern) for _ in range(clone_num)])
            clones = np.unique(clones, axis=0)
            clones = clones[np.any(clones != pop[arg], axis=1)]
            if clones.shape[0] > 0:
                if self.repair_rpc is not None:
                    clones = self.evaluate_call(clones, self.repair_rpc)
                clones_eval = self.evaluate_call(clones, self.obj_rpc)
                evaluation_counter += clones.shape[0]
                argmin = clones_eval.argmin()
                if clones_eval[argmin] < temp_pop_eval[arg]:
                    indices.append(arg)
                    better.append(clones[argmin])
                    better_eval.append(clones_eval[argmin])
        if len(better) > 0:
            better = np.stack(better)
            better_eval = np.stack(better_eval)
            temp_pop[indices] = better
            temp_pop_eval[indices] = better_eval
        return temp_pop, temp_pop_eval, evaluation_counter

    def _mutate(self, ind, pattern):
        a, b, sigma = self.mutate_args
        r = np.random.random()
        if r < a:
            ind = self._uniform_mutate(ind, pattern, self.bounds)
        elif r < b:
            ind = self._gaussian_mutate(ind, pattern, self.bounds, sigma)
        else:
            ind = self._bound_mutate(ind, pattern, self.bounds)
        return ind

    @staticmethod
    def _uniform_mutate(individual, pattern, bounds):
        ind = individual.copy()
        if np.sum(pattern) == 0:
            return ind
        indx = np.where(pattern)[0]
        k = np.random.choice(indx)
        a = bounds[k][0]
        b = bounds[k][1]
        ind[k] = np.random.uniform(a, b)
        return ind

    @staticmethod
    def _bound_mutate(individual, pattern, bounds):
        ind = individual.copy()
        if np.sum(pattern) == 0:
            return ind
        indx = np.where(pattern)[0]
        k = np.random.choice(indx)
        a = bounds[k][0]
        b = bounds[k][1]
        r = np.random.random()
        r2 = np.random.uniform(0, 1)
        if r < 0.5:
            ind[k] = a + (ind[k] - a) * r2
        else:
            ind[k] = (b - ind[k]) * r2 + ind[k]
        return ind

    @staticmethod
    def _gaussian_mutate(individual, pattern, bounds, sigma):
        ind = individual.copy()
        if np.sum(pattern) == 0:
            return ind
        indx = np.where(pattern)[0]
        k = np.random.choice(indx)
        a = bounds[k][0]
        b = bounds[k][1]
        ran = sigma * (b - a) * np.random.randn() + ind[k]
        if a <= ran <= b:
            ind[k] = ran
        return ind