from osbrain import run_nameserver
from osbrain import run_agent
from pyagamo.utils import assigning_gens, get_not_dominated, front_suppression
from pyagamo.utils import PopMemoryQueue
from copy import deepcopy
from multiprocessing import Manager
import numpy as np
from tqdm import tqdm
import asyncio
import threading


class AGAMO:
    def __init__(self, max_eval, change_iter, next_iter, max_front, init_pop='separate', mq=0, ns=None, transport='ipc',
                 verbose=False):
        self.max_front = max_front
        self.max_eval = max_eval
        self.change_iter = change_iter
        self.next_iter = next_iter
        self.init_pop = init_pop
        self.ns = ns
        if ns is None:
            self.ns = run_nameserver()
        self.transport = transport
        self.verbose = verbose
        self.repair = None
        self.repair_addr = None
        self.mq = mq
        self.qpop = None
        if self.mq > 0:
            self.qpop = PopMemoryQueue(mq)
        
        try:
            manager = getattr(type(self), 'manager')
        except AttributeError:
            manager = type(self).manager = Manager()
        self._lock = manager.RLock()
       
        self._shared_best = manager.dict({})
        self._shared_front = manager.dict({})
        self._shared_values = manager.dict({})
        
    def add_objectives(self, objectives, objectives_m=None):
        self.objectives = objectives
        self.nobjs = objectives[0].n_obj
        self.nvars = objectives[0].n_var
        self.bounds = objectives[0].bounds
        self.objectives_m = objectives_m
                    
    def add_repair(self, repair):
        self.repair = repair
                    
    def add_players(self, players):
        self.players = players
    
    def init(self):
        # kopia objectives
        if self.objectives_m is None:
            self.objectives_m = deepcopy(self.objectives)
            for obj in self.objectives_m:
                obj.num = obj.num + len(self.objectives_m)
        
        self.objs_addr = []
        for obj in self.objectives:
            self.objs_addr.append(obj.run(ns=self.ns))
            
        self.objs_m_addr = []
        for obj in self.objectives_m:
            self.objs_m_addr.append(obj.run(ns=self.ns))
        
        if self.repair is not None:
            self.repair_addr = self.repair.run(ns=self.ns)
        
        self.best_agent = run_agent('best', self.ns.addr(), transport=self.transport)
        self.best_addr = self.best_agent.bind('REP', alias='get_set_best', handler=lambda a, m: self._reply_best(a, m),
                                              transport=self.transport)
        
        self.mbest_agent = run_agent(f'best_f', self.ns.addr(), transport=self.transport)
        self.mbest_agent.connect(self.best_addr, alias='get_set_best')
        
        self.cmd_agent = run_agent('cmd_pub', self.ns.addr(), transport=self.transport)
        self.cmd_addr = self.cmd_agent.bind('PUB', alias='command')
        
        self.obj_agents = []
        for i, addr in enumerate(self.objs_m_addr):
            self.obj_agents.append(run_agent(f'f_call_p{i}', self.ns.addr(), transport=self.transport))
            self.obj_agents[-1].connect(addr, alias='evaluate')
        
        self.front_agent = run_agent('front', self.ns.addr(), transport=self.transport)
        self.front_addr = self.front_agent.bind('ASYNC_REP', alias='push_front',
                                                handler=lambda a, m: self._pop_consumer(a, m), transport=self.transport)
        
        self.player_processes = []
        for i, player in enumerate(self.players):
            self.player_processes.append(player.run(self.objs_addr[i], self.repair_addr, self.best_addr, self.cmd_addr,
                                                    self.front_addr, self.nvars, self.bounds, self.next_iter,
                                                    ns=self.ns))

    def start_optimize(self, tqdm_disable=False, verbose=1, thread=False):
        if thread:
            t1 = threading.Thread(target=self._start_optimize, args=(True, verbose))
            t1.start()
        else:
            self._start_optimize(tqdm_disable=tqdm_disable, verbose=verbose)

    def _start_optimize(self, tqdm_disable=False, verbose=1):
        first = True
        self._shared_best['solutions'] = [None] * self.nobjs
        self._shared_best['iter_counters'] = [None] * self.nobjs
        self._shared_best['front'] = None
        self._shared_best['front_eval'] = None 
        self._shared_front['stop_flag'] = False
        self._shared_front['front'] = []
        self._shared_front['front_eval'] = []
        self._shared_front['iterations'] = []
        self._shared_front['evaluations'] = []
        self._shared_front['evaluations_m'] = []
        self._shared_front['nobjs'] = self.nobjs
        self._shared_front['max_front'] = self.max_front
        self._shared_front['change_iter'] = self.change_iter
        self._shared_front['max_eval'] = self.max_eval
        self._shared_front['min_iter_pop'] = 0
        self._shared_front['change_flag'] = True
        start_flag = True
        
        with tqdm(total=self.max_eval, unit='eval', disable=tqdm_disable) as pbar:
            while start_flag:
                if self._shared_front['stop_flag']:
                    start_flag = not self._shared_front['stop_flag']
                if start_flag:
                    if first:
                        first = False
                        with self._lock:
                            self._shared_front['front'] = []
                            self._shared_front['front_eval'] = []
                            self._shared_front['evaluations'] = []
                            self._shared_front['evaluations_m'] = []
                            self._shared_front['iterations'] = []
                            self._shared_front['min_iter_pop'] = 0
                            self._shared_front['change_flag'] = True
                            self._shared_front['stop_flag'] = False
                        # sending parms to players
                        # sending pop
                        if self.init_pop == 'same':
                            pop = self.players[0].create_population()
                            msg = ['parm', 'Pop', pop]
                            self.cmd_agent.send('command', msg)
                        # sending gens
                        if self._shared_front['change_flag']:
                            patterns = assigning_gens(self.nvars, self.nobjs)
                            msg = ['parm', 'Gens', patterns]
                            self.cmd_agent.send('command', msg)
                            with self._lock:
                                self._shared_front['change_flag'] = False
                        # sending START to players
                        msg = ['cmd', 'Start']
                        self.cmd_agent.send('command', msg)
                        continue
                    # sending gens
                    if self._shared_front['change_flag']:
                        patterns = assigning_gens(self.nvars, self.nobjs)
                        msg = ['parm', 'Gens', patterns]
                        self.cmd_agent.send('command', msg)
                        with self._lock:
                            self._shared_front['change_flag'] = False
                else:
                    if first:
                        continue
                    else:
                        # sending STOP to players
                        msg = ['cmd', 'Stop']
                        self.cmd_agent.send('command', msg)
                        first = True
                        with self._lock:
                            self._shared_front['change_flag'] = True
                
                e = self._shared_front.get('evaluations')
                em = self._shared_front.get('evaluations_m')
                if isinstance(e, np.ndarray) and isinstance(em, np.ndarray):
                    evaluations = np.min(e+em)
                    pbar.update(evaluations-pbar.n)
        
        for p in self.player_processes:
            p.terminate()

    def get_results(self):
        res = deepcopy(self._shared_front)
        if 'nobjs' in res:
            del res['nobjs']
            del res['max_front']
            del res['change_iter']
            del res['max_eval']
            del res['min_iter_pop']
            del res['change_flag']
        return res
    
    def close(self):
        self.ns.shutdown()
    
    def _reply_best(self, agent, message):
        player_data = message
        nobj = player_data.get('nobj', None)
        if nobj is not None:
            solution = player_data['population']
            iterr = player_data['iter_counter']
            solutions = deepcopy(self._shared_best['solutions'])
            iterrs = deepcopy(self._shared_best['iter_counters'])
            solutions[nobj] = solution
            iterrs[nobj] = iterr
            with self._lock:
                self._shared_best['solutions'] = solutions
                self._shared_best['iter_counters'] = iterrs
        else:
            front = player_data['front']
            front_eval = player_data['front_eval']
            with self._lock:
                self._shared_best['front'] = front
                self._shared_best['front_eval'] = front_eval
        
        best = [deepcopy(self._shared_best['solutions']), deepcopy(self._shared_best['iter_counters']),
                deepcopy(self._shared_best['front']), deepcopy(self._shared_best['front_eval'])]
        return best
    
    def _pop_consumer(self, agent, message):
        front = self._shared_front['front']
        front_eval = self._shared_front['front_eval']
        evaluations = self._shared_front['evaluations']
        evaluations_m = self._shared_front['evaluations_m']
        iterations = self._shared_front['iterations']
        max_front = self._shared_front['max_front']
        change_flag = self._shared_front['change_flag']
        change_iter = self._shared_front['change_iter']
        min_iter_pop = self._shared_front['min_iter_pop']
        stop_flag = self._shared_front['stop_flag']
        max_eval = self._shared_front['max_eval']
        nobjs = self._shared_front['nobjs']

        if stop_flag:
            return None

        player_data = message
        nobj = player_data['nobj']
        pop = player_data['population']
        pop_eval = player_data['population_eval']
        evaluation_counter = player_data['evaluation_counter']
        iteration = player_data['iteration']

        if len(iterations) == 0:
            iterations = np.zeros(nobjs)
        iterations[nobj] = iteration

        min_iter = np.min(iterations)
        if min_iter - min_iter_pop >= change_iter:
            change_flag = True
            min_iter_pop = min_iter
            
        with self._lock:
            self._shared_front['iterations'] = iterations
            self._shared_front['change_flag'] = change_flag
            self._shared_front['min_iter_pop'] = min_iter_pop

        if len(evaluations) == 0:
            evaluations = np.zeros(nobjs)
        evaluations[nobj] = evaluation_counter

        pop_evals = []
        if len(evaluations_m) == 0:
            evaluations_m = np.zeros(nobjs)
        
        if self.qpop is not None:
            pop, pop_eval, pop_old, pop_old_evals = self.qpop.get_new(pop, pop_eval, nobj)
        else:
            pop, pop_eval, pop_old, pop_old_evals = pop, pop_eval, None, None
       # agent.log_info(f'{agent.name}: shape: {pop.shape}')
            
        if pop.shape[0] > 0:
            for i in range(nobjs):
                if i != nobj:
                    self.obj_agents[i].send('evaluate', pop)
                    pop_evals.append(np.reshape(self.obj_agents[i].recv('evaluate'), (-1, 1)))
                    evaluations_m[i] += pop.shape[0]
                else:
                    pop_evals.append(np.reshape(pop_eval, (-1, 1)))
            pop_evals = np.hstack(pop_evals)
            if self.qpop is not None:
                self.qpop.add_new(pop, pop_evals)
        
        if pop_old is not None:
            if pop.shape[0] > 0:
                pop = np.concatenate((pop, pop_old))
                pop_evals = np.concatenate((pop_evals, pop_old_evals))
            else:
                pop = pop_old
                pop_evals = pop_old_evals
       # agent.log_info(f'{agent.name}: shape: {self.qpop.pop_queue.shape}')
                                   
        if max_eval > 0:
            if np.min(evaluations + evaluations_m) >= max_eval:
                stop_flag = True
                msg = ['cmd', 'Stop']
                self.cmd_agent.send('command', msg)
                
        with self._lock:
            self._shared_front['evaluations'] = evaluations
            self._shared_front['evaluations_m'] = evaluations_m
            self._shared_front['stop_flag'] = stop_flag
            
        if pop.shape[0] > 1:
            unique_pop_eval, indices = np.unique(pop_evals, axis=0, return_index=True)
            pop = pop[indices,:]
            pop_evals = pop_evals[indices]
                
        if pop.shape[0] > 1:
            mask = get_not_dominated(pop_evals)
            pop = pop[mask]
            pop_evals = pop_evals[mask]
            
        if pop.shape[0] > 0:
            if len(front) == 0:
                front = deepcopy(pop)
                front_eval = deepcopy(pop_evals)
            else:
                front = np.vstack([front, pop])
                front_eval = np.vstack([front_eval, pop_evals])
                unique_front_eval, indices = np.unique(front_eval, axis=0, return_index=True)
                front = front[indices,:]
                front_eval = front_eval[indices]
                mask = get_not_dominated(front_eval)
                front = front[mask]
                front_eval = front_eval[mask]

        # front suppression
        if len(front) > 0 and (0 < max_front < front.shape[0]):
            mask = front_suppression(front_eval, max_front)
            front = front[mask]
            front_eval = front_eval[mask]  
        
        if len(front) > 0:
            sfront = deepcopy(front)
            sfront_eval = deepcopy(front_eval)
            shared_best = {'front': sfront, 'front_eval': sfront_eval}
            self.mbest_agent.send('get_set_best', shared_best)
            res = self.mbest_agent.recv('get_set_best')
                                   
        with self._lock:
            self._shared_front['front'] = front
            self._shared_front['front_eval'] = front_eval
