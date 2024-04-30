import numpy as np
from pyagamo import Player
from copy import deepcopy
from scipy.stats import rankdata


class ClonalSelection(Player):
    def __init__(self, num, npop, player_parm, mq=0, ns=None, transport='ipc', verbose=False):
        self.nclone = player_parm.get('nclone', 15)
        self.mutate_args = tuple(player_parm.get('mutate_args', [0.45, 0.9, 0.01]))
        self.sup = player_parm.get('sup', 0.0)
        super(ClonalSelection, self).__init__(num, npop, mq=mq, ns=ns, transport=transport, verbose=verbose)

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
            clones = clones[np.any(clones != temp_pop[arg], axis=1)]
    
            if clones.shape[0] > 0:
                if self.qclone is not None:
                    clones, clones_old, clones_old_eval = self.qclone.get_new(clones)
                else:
                    clones, clones_old, clones_old_eval = clones, None, None
                if clones.shape[0] > 0:
                    clones = self.repair_call(clones)
                    clones_eval = self.evaluate_call(clones)
                    evaluation_counter += clones.shape[0]
                    if self.qclone is not None:
                        self.qclone.add_new(clones, clones_eval)
                    if clones_old is not None:
                        clones = np.concatenate((clones, clones_old))
                        clones_eval = np.concatenate((clones_eval, clones_old_eval))
                elif clones_old is not None:
                    clones = clones_old.copy()
                    clones_eval = clones_old_eval.copy()
                    
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
            
        arg_sort = temp_pop_eval.argsort()
        d = int(pop.shape[0]*self.sup)
        if d > 0:
            inds = temp_pop_eval.argsort()[-d:]
            pop_sup = np.zeros((inds.shape[0], self.nvars))
            for i in range(inds.shape[0]):
                pop_sup[i] = pop_sup[i] + np.where(pattern, self._create_individual_uniform(self.bounds), temp_pop[inds[i]])
            pop_sup = self.repair_call(pop_sup)
            pop_eval_sup = self.evaluate_call(pop_sup)
            evaluation_counter += pop_sup.shape[0]
            temp_pop[inds,:] = pop_sup[:,:]
            temp_pop_eval[inds] = pop_eval_sup[:]
            
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
        s = np.sum(pattern)
        if s == 0:
            return ind
        r = np.random.random(pattern.shape) < 1/s
        r = np.logical_and(pattern, r)
        indx = np.where(r)[0]
        if len(indx)>0:
            for k in indx:
                a = bounds[k][0]
                b = bounds[k][1]
                ind[k] = np.random.uniform(a, b)
        else:
            indx = np.where(pattern)[0]
            k = np.random.choice(indx)
            a = bounds[k][0]
            b = bounds[k][1]
            ind[k] = np.random.uniform(a, b)
        return ind

    @staticmethod
    def _bound_mutate(individual, pattern, bounds):
        ind = individual.copy()
        s = np.sum(pattern)
        if s == 0:
            return ind
        r = np.random.random(pattern.shape) < 1/s
        r = np.logical_and(pattern, r)
        indx = np.where(r)[0]
        if len(indx)>0:
            for k in indx:
                a = bounds[k][0]
                b = bounds[k][1]
                r1 = np.random.random()
                r2 = np.random.uniform(0, 1)
                if r1 < 0.5:
                    ind[k] = a + (ind[k] - a) * r2
                else:
                    ind[k] = (b - ind[k]) * r2 + ind[k]
        else:
            indx = np.where(pattern)[0]
            k = np.random.choice(indx)
            a = bounds[k][0]
            b = bounds[k][1]
            r1 = np.random.random()
            r2 = np.random.uniform(0, 1)
            if r1 < 0.5:
                ind[k] = a + (ind[k] - a) * r2
            else:
                ind[k] = (b - ind[k]) * r2 + ind[k]
        return ind

    @staticmethod
    def _gaussian_mutate(individual, pattern, bounds, sigma):
        ind = individual.copy()
        s = np.sum(pattern)
        if s == 0:
            return ind
        r = np.random.random(pattern.shape) < 1/s
        r = np.logical_and(pattern, r)
        indx = np.where(r)[0]
        if len(indx)>0:
            for k in indx:
                a = bounds[k][0]
                b = bounds[k][1]
                ran = sigma * (b - a) * np.random.randn() + ind[k]
                if a <= ran <= b:
                    ind[k] = ran
                elif ran < a:
                    ind[k] = a
                else:
                    ind[k] = b
        else:
            indx = np.where(pattern)[0]
            k = np.random.choice(indx)
            a = bounds[k][0]
            b = bounds[k][1]
            ran = sigma * (b - a) * np.random.randn() + ind[k]
            if a <= ran <= b:
                ind[k] = ran
            elif ran < a:
                ind[k] = a
            else:
                ind[k] = b
        return ind
