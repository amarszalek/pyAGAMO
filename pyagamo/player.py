from osbrain import run_agent
from osbrain import run_nameserver
from copy import deepcopy
from multiprocessing import Process, Manager
import numpy as np
from pyagamo.utils import front_suppression, get_not_dominated
from pyagamo.utils import CloneMemoryQueue

class Player:
    def __init__(self, num, npop, mq=0, ns=None, transport='ipc', verbose=False):
        self.num = num
        self.npop = npop
        self.ns = ns
        self.transport = transport
        self.repair = False
        self.verbose = verbose
        self.mq = mq
        self.qclone = None
        if mq > 0:
            self.qclone = CloneMemoryQueue(mq)

        try:
            manager = getattr(type(self), 'manager')
        except AttributeError:
            manager = type(self).manager = Manager()
        self._shared_values = manager.dict({})
        self._shared_values['start_flag'] = False
        self._shared_values['pop'] = None
        self._lock = manager.RLock()
        
    def step(self, pop, pop_eval, pattern):
        raise NotImplementedError('You must override this method in your class!')
        
    def create_population(self):
        pop = np.zeros((self.npop, self.nvars))
        for i in range(self.npop):
            pop[i] = self._create_individual_uniform(self.bounds)
        return pop
    
    @staticmethod
    def _create_individual_uniform(bounds):
        a = np.array([bounds[k][0] for k in range(len(bounds))])
        b = np.array([bounds[k][1] for k in range(len(bounds))])
        return np.random.uniform(a, b)
        
    def run(self, obj_addr, repair_addr, best_addr, cmd_addr, front_addr, nvars, bounds, next_iter, ns=None):
        self.nvars = nvars
        self.bounds = bounds
        self.next_iter = next_iter
        if ns is None:
            self.ns = run_nameserver()
        else:
            self.ns = ns
        # agent do wywoływania funkcji celu    
        self.call_agent = run_agent(f'call_p{self.num}', self.ns.addr(), transport=self.transport)
        self.call_agent.connect(obj_addr, alias='evaluate')
        
        # agent do wywoływania funkcji naprawy
        if repair_addr is not None:
            self.rep_agent = run_agent(f'rep_p{self.num}', self.ns.addr(), transport=self.transport)
            self.rep_agent.connect(repair_addr, alias='repair')
            self.repair = True
            
        # agent do odczytywania poleceń
        self.cmd_agent = run_agent(f'cmd_p{self.num}', self.ns.addr(), transport=self.transport)
        self.cmd_agent.connect(cmd_addr, handler=lambda a,m: self._cmd_consumer(a,m))
        
        # agent do wysyłania populacji i odbierania frontu
        self.mbest_agent = run_agent(f'best_p{self.num}', self.ns.addr(), transport=self.transport)
        self.mbest_agent.connect(best_addr, alias='get_set_best')

        # agent do wysyłania populacji na front
        self.front_agent = run_agent(f'front_p{self.num}', self.ns.addr(), transport=self.transport)
        self.front_agent.connect(front_addr, alias='push_front', handler=lambda a, m: None)
        
        p1 = Process(target=self._main_process, args=(self,))
        p1.daemon = True
        p1.start()
        
        return p1
    
    def evaluate_call(self, solutions):
        self.call_agent.send('evaluate', solutions)
        return self.call_agent.recv('evaluate')
    
    def best_call(self, shared_best):
        self.mbest_agent.send('get_set_best', shared_best)
        return self.mbest_agent.recv('get_set_best')
    
    def front_call(self, shared_population):
        self.front_agent.send('push_front', shared_population)
    
    def repair_call(self, solutions):
        if self.repair:
            self.rep_agent.send('repair', solutions)
            return self.rep_agent.recv('repair')
        else:
            return solutions
        
    def _cmd_consumer(self, agent, msg):
        if self.verbose:
            agent.log_info(f'{agent.name}: cmd: {msg}')
            
        if msg[0] == 'cmd':
            if msg[1] == 'Start':
                with self._lock:
                    self._shared_values['start_flag'] = True
            elif msg[1] == 'Stop':
                with self._lock:
                    self._shared_values['start_flag'] = False
        elif msg[0] == 'parm':
            if msg[1] == 'Gens':
                with self._lock:
                    self._shared_values['patterns'] = msg[2]
            if msg[1] == 'Pop':
                with self._lock:
                    self._shared_values['pop'] = msg[2]
                    
    @staticmethod
    def _main_process(self):
        obj = self.num
        shared_best = {'nobj': obj, 'population': None, 'iter_counter': 0}
        shared_population = {'nobj': obj, 'iteration': 0}

        first = True
        pop = None
        pop_eval = None
        evaluation_counter = 0
        next_iter_counter = 0
        iters_pop = None
        bests = None
        bests_eval = None
        while True: 
            if self._shared_values['start_flag']:
                patterns = deepcopy(self._shared_values['patterns'])
                pattern = patterns[obj]
                if first:
                    iters_pop = None
                    next_iter_counter = 0
                    shared_population['evaluation_counter'] = 0
                    shared_population['iteration'] = 0

                    # create and evaluate population
                    if self._shared_values['pop'] is None:
                        pop = self.create_population()
                    else:
                        pop = self._shared_values['pop']
                    
                    pop = self.repair_call(pop)
                    pop_eval = self.evaluate_call(pop)
                    evaluation_counter = pop.shape[0]
                    shared_population['evaluation_counter'] = evaluation_counter
                    first = False

                # optimize step
                if self.next_iter <= 0 or self.next_iter - next_iter_counter > 0:
                    if pattern.sum() > 0:
                        pop, pop_eval, neval = self.step(pop, pop_eval, pattern)
                        evaluation_counter += neval
                    shared_population['iteration'] += 1
                    shared_population['evaluation_counter'] = evaluation_counter
                    
                    if self._shared_values['start_flag']:# and pattern.sum() > 0:
                        unique_pop, indices = np.unique(pop, axis=0, return_index=True)
                        unique_pop_eval = pop_eval[indices]
                        shared_population['population'] = deepcopy(unique_pop)
                        shared_population['population_eval'] = deepcopy(unique_pop_eval)
                    
                    # send and recv best
                    shared_best['population'] = (deepcopy(pop), deepcopy(pop_eval))
                    shared_best['iter_counter'] = shared_population['iteration']
                    res = self.best_call(shared_best)
                    
                    front = res[2]
                    front_eval = res[3]
                    if front is not None and len(front) > 0:
                        # front suppression
                        arr = np.arange(front.shape[0])
                        np.random.shuffle(arr)
                        front = front[arr]
                        front_eval = front_eval[arr]
                            
                        if pop.shape[0] < front.shape[0]:
                            mask = front_suppression(front_eval, pop.shape[0])
                            front = front[mask]
                            front_eval = front_eval[mask]
                            
                        nn = pop.shape[0]
                        inds = np.random.choice(front.shape[0], nn, replace=True)
                        arg_sort = pop_eval.argsort()
                        for i, j in enumerate(arg_sort[-nn:]):
                            pop[j, np.logical_not(pattern)] = front[inds[i], np.logical_not(pattern)]
                        
                        if self.qclone is not None:
                            clones, clones_old, clones_old_eval = self.qclone.get_new(pop)
                        else:
                            clones, clones_old, clones_old_eval = pop, None, None
                        if clones.shape[0] > 0:
                            clones = self.repair_call(clones)
                            clones_eval = self.evaluate_call(clones)
                            evaluation_counter += clones.shape[0]
                            if self.qclone is not None:
                                self.qclone.add_new(clones, clones_eval)
                        else:
                            clones_eval = np.array([])
                        if clones_old is not None:
                            pop = np.concatenate((clones, clones_old))
                            pop_eval = np.concatenate((clones_eval, clones_old_eval))
                        else:
                            pop = clones.copy()
                            pop_eval = clones_eval.copy()
                        shared_population['evaluation_counter'] = evaluation_counter
                   
                    next_iter_counter += 1
                                
                res = self.best_call(shared_best)
                iters = res[1]
                iters[obj] = shared_population['iteration']
                nobjs = len(iters)
                iters_mask = np.zeros(nobjs, dtype=bool)
                for i in range(nobjs):
                    if iters_pop is None or iters_pop[i] != iters[i]:
                        iters_mask[i] = True
   
                if np.all(iters_mask[:obj]) and np.all(iters_mask[obj + 1:]):
                    self.front_call(shared_population)
                    next_iter_counter = 0
                    iters_pop = deepcopy(iters)

            else:
                first = True
                next_iter_counter = 0
                shared_best['population'] = None
                shared_best['iter_counter'] = 0

