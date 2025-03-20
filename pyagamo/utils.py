import numpy as np
from copy import deepcopy

CEXT = False
try:
    from pyagamo.cutils import cget_not_dominated, cfront_suppression
    CEXT = True
except Exception as e:
    print('C extension not available')
    print(e)


def assigning_gens(nvars, nobjs):
    while True:
        if nvars <= nobjs:
            #r = np.random.choice(range(nobjs), size=(nvars,), replace=False)
            #r2 = np.stack([r == i for i in range(nobjs)])
            r = np.random.choice(range(nvars), size=(nobjs,), replace=True)
            r2 = np.stack([r == i for i in range(nvars)])
            r2 = r2.T
            break
        else:
            r = np.random.randint(0, nobjs, size=(nvars,))
            r2 = np.stack([r == i for i in range(nobjs)])
            if nvars >= nobjs and not np.any(np.all(r2, axis=1)) and not np.any(np.all(np.logical_not(r2), axis=1)):
                break
    return r2


def pairwise_dominance(x):
    z = x[:, np.newaxis] >= x #org
    z = np.all(z, axis=2)
    z[range(z.shape[0]), range(z.shape[0])] = False
    #z[np.triu_indices(z.shape[0])] = False
    xx = np.any(z, axis=1)
    return np.logical_not(xx)


def get_not_dominated(populations_eval):
    if CEXT:
        mask = np.zeros(populations_eval.shape[0], dtype=np.int32)
        cget_not_dominated(populations_eval, mask)
    else:
        mask = pairwise_dominance(populations_eval)
    return mask


def pairwise_distance(x):
    return np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)


def front_suppression(front_eval, front_max):
    if CEXT:
        mask = np.zeros(front_eval.shape[0], dtype=np.int32)
        cfront_suppression(front_eval, front_max, mask)
    else:
        n = front_eval.shape[0] - front_max
        ideal = np.argmin(front_eval, axis=0)
        front_eval_norm = front_eval + np.abs(np.min(front_eval, axis=0))+1.0
        front_eval_norm = front_eval_norm/np.max(front_eval_norm, axis=0)
        z = pairwise_distance(front_eval_norm)
        mask = np.ones(front_eval.shape[0], dtype=bool)
        t = np.tril(z) + np.triu(np.ones_like(z) * 1000000)
        arg = np.argsort(t, axis=None)
        indx_i, indx_j = np.unravel_index(arg, t.shape)
        while n > 0:
            ii = indx_i[0]
            mask[ii] = False
            tmp = indx_i[indx_i!=ii]
            indx_j = indx_j[indx_i!=ii]
            indx_i = tmp.copy()
            tmp = indx_j[indx_j!=ii]
            indx_i = indx_i[indx_j!=ii]
            indx_j = tmp.copy()
            n=n-1
        for i in ideal:
            mask[i] = True
    return mask


class PopMemoryQueue():
    def __init__(self, max_len):
        self.max_len = max_len
        self.pop_queue = None
        self.pop_eval_queue = None
        
    def get_new(self, pop, pop_eval, obj):
        pop, indices = np.unique(pop, axis=0, return_index=True)
        pop_eval = pop_eval[indices]
        if self.pop_queue is None:
            return pop, pop_eval, None, None
        else:
            p = self.pop_queue[:, np.newaxis] == pop
            p = np.all(p, axis=2)
            ind_x, ind_y = np.where(p) 
            xx = np.any(p, axis=0)
            mask = np.logical_not(xx)
            return pop[mask], pop_eval[mask], self.pop_queue[ind_x], self.pop_eval_queue[ind_x]
    
    def add_new(self, pop, pop_eval):
        if self.pop_queue is None:
            self.pop_queue = deepcopy(pop[:self.max_len])
            self.pop_eval_queue = deepcopy(pop_eval[:self.max_len])
        else:
            tmp = np.concatenate((self.pop_queue, pop))
            tmp_eval = np.concatenate((self.pop_eval_queue, pop_eval))
            #tmp, indices = np.unique(tmp, axis=0, return_index=True)
            #tmp_eval = tmp_eval[indices]
            self.pop_queue = deepcopy(tmp[-self.max_len:])
            self.pop_eval_queue = deepcopy(tmp_eval[-self.max_len:])
            

class CloneMemoryQueue():
    def __init__(self, max_len):
        self.max_len = max_len
        self.pop_queue = None
        self.pop_eval_queue = None
        
    def get_new(self, pop):
        #pop, indices = np.unique(pop, axis=0, return_index=True)
        #pop_eval = pop_eval[indices]
        if self.pop_queue is None:
            return pop, None, None
        else:
            p = self.pop_queue[:, np.newaxis] == pop
            p = np.all(p, axis=2)
            ind_x, ind_y = np.where(p) 
            xx = np.any(p, axis=0)
            mask = np.logical_not(xx)
            return pop[mask], self.pop_queue[ind_x], self.pop_eval_queue[ind_x]
    
    def add_new(self, pop, pop_eval):
        pop, indices = np.unique(pop, axis=0, return_index=True)
        pop_eval = pop_eval[indices]
        if self.pop_queue is None:
            self.pop_queue = deepcopy(pop[:self.max_len])
            self.pop_eval_queue = deepcopy(pop_eval[:self.max_len])
        else:
            tmp = np.concatenate((self.pop_queue, pop))
            tmp_eval = np.concatenate((self.pop_eval_queue, pop_eval))
            #tmp, indices = np.unique(tmp, axis=0, return_index=True)
            #tmp_eval = tmp_eval[indices]
            self.pop_queue = deepcopy(tmp[-self.max_len:])
            self.pop_eval_queue = deepcopy(tmp_eval[-self.max_len:])
            
