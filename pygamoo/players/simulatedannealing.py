import numpy as np
from pygamoo import Player
from copy import deepcopy


class SimulatedAnnealing(Player):
    def __init__(self, num, obj_queue, repair_queue, cmd_exchange, npop, nvars, bounds, host, port, player_parm):
        self.temp = player_parm['temp']
        self.dec_step = player_parm['dec_step']
        self.mutate_args = tuple(player_parm['mutate_args'])
        super(SimulatedAnnealing, self).__init__(num, obj_queue, repair_queue, cmd_exchange, npop, nvars, bounds, host, port)

    def step(self, pop, pop_eval, pattern):
        temp_pop = deepcopy(pop)
        temp_pop_eval = deepcopy(pop_eval)
        evaluation_counter = 0
        args = (pattern,)
        clones = np.apply_along_axis(self._mutate, 1, temp_pop, *args)
        if self.repair_rpc is not None:
            clones = self.evaluate_call(clones, self.repair_rpc)
        clones_eval = self.evaluate_call(clones, self.obj_rpc)
        evaluation_counter += clones.shape[0]
        r = np.random.random(size=self.npop)
        p = np.exp((temp_pop_eval - clones_eval) / self.temp)
        mask = r < p
        temp_pop[mask, :] = clones[mask, :]
        temp_pop_eval[mask] = clones_eval[mask]
        self.temp = self.temp * self.dec_step
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
