import numpy as np
import pickle
import pika
from pygamoo.utils import RpcClient, evaluate_call
from multiprocessing import Process, Manager
from copy import deepcopy


class Player:
    def __init__(self, num, obj_queue, repair_queue, cmd_exchange, npop, nvars, bounds, host, port):
        self.num = num
        self.obj_queue = obj_queue
        self.repair_queue = repair_queue
        self.cmd_exchange = cmd_exchange
        self.npop = npop
        self.nvars = nvars
        self.bounds = bounds
        self.host = host
        self.port = port
        self._p1 = None
        self._p2 = None
        try:
            manager = getattr(type(self), 'manager')
        except AttributeError:
            manager = type(self).manager = Manager()
        self._shared_values = manager.dict({})
        self._shared_values['start_flag'] = False
        self._lock = manager.RLock()

    def step(self, pop, pop_eval, pattern):
        raise NotImplementedError('You must override this method in your class!')

    def create_population(self):
        pop = np.zeros((self.npop, self.nvars))
        for i in range(self.npop):
            pop[i] = self._create_individual_uniform(self.bounds)
        return pop

    @staticmethod
    def evaluate_call(solutions, rpc):
        res = rpc.call(pickle.dumps(solutions))
        return pickle.loads(res, encoding='bytes')

    def run(self):
        p1 = Process(target=self._cmd_consumer, args=(self,))
        p1.daemon = True
        p1.start()
        p2 = Process(target=self._main_process, args=(self,))
        p2.daemon = True
        p2.start()
        self._p1 = p1
        self._p2 = p2

    def is_alive(self, separate=False):
        if separate:
            return (False if self._p1 is None else self._p1.is_alive(),
                    False if self._p2 is None else self._p2.is_alive())
        if (self._p1 is not None) and (self._p2 is not None):
            return all([self._p1.is_alive(), self._p2.is_alive()])
        return False

    def is_working(self):
        return self._shared_values['start_flag']

    def _start(self):
        with self._lock:
            self._shared_values['start_flag'] = True

    def _stop(self):
        with self._lock:
            self._shared_values['start_flag'] = False

    def close(self):
        if self._p1 is not None:
            self._p1.terminate()
        if self._p2 is not None:
            self._p2.terminate()
        #connection = pika.BlockingConnection(pika.ConnectionParameters(host=self.host, port=self.port))
        #channel = connection.channel()
        #channel.exchange_delete(exchange=self.cmd_exchange)

    def __del__(self):
        self.close()

    @staticmethod
    def _main_process(self):
        obj = self.num
        shared_best = {'nobj': obj, 'solution': None, 'iter_counter': 0}
        shared_population = {'nobj': obj, 'iteration': 0}

        self.obj_rpc = RpcClient(self.obj_queue, self.host, self.port)
        self.repair_rpc = None
        if self.repair_queue is not None:
            self.repair_rpc = RpcClient(self.repair_queue, self.host, self.port)

        first = True
        pop = None
        pop_eval = None
        best_rpc = None
        evaluation_counter = 0
        next_iter_counter = 0
        best_solutions_pop = None
        iters_pop = None
        while True:
            if self._shared_values['start_flag']:
                patterns = deepcopy(self._shared_values['patterns'])
                pattern = patterns[obj]
                if first:
                    best_solutions_pop = None
                    iters_pop = None
                    next_iter_counter = 0
                    shared_population['evaluation_counter'] = 0
                    shared_population['iteration'] = 0
                    best_rpc = RpcClient(self._shared_values['manager_best_pull_queue'], self.host, self.port)

                    # create and evaluate population
                    pop = self.create_population()
                    if self.repair_rpc is not None:
                        pop = evaluate_call(pop, self.repair_rpc)
                    pop_eval = evaluate_call(pop, self.obj_rpc)
                    evaluation_counter = pop.shape[0]

                    # found best
                    argmin = pop_eval.argmin()
                    best_solution = pop[argmin]
                    if self._shared_values['start_flag']:
                        shared_best['solution'] = deepcopy(best_solution)
                        shared_best['iter_counter'] = shared_population['iteration']
                        _send_to(self._shared_values['manager_best_push_queue'], shared_best, self.host, self.port)
                        shared_population['iteration'] = 0
                        shared_population['population'] = deepcopy(pop)
                        shared_population['population_eval'] = deepcopy(pop_eval)
                        shared_population['evaluation_counter'] = evaluation_counter
                        _send_to(self._shared_values['manager_pop_queue'], shared_population, self.host, self.port)
                    first = False

                # optimize step
                if self._shared_values['next_iter'] <= 0 or self._shared_values['next_iter'] - next_iter_counter > 0:
                    pop, pop_eval, neval = self.step(pop, pop_eval, pattern)
                    evaluation_counter += neval
                    # getting best
                    argmin = pop_eval.argmin()
                    best_solution = pop[argmin]
                    shared_population['iteration'] += 1
                    shared_best['solution'] = deepcopy(best_solution)
                    shared_best['iter_counter'] = shared_population['iteration']
                    _send_to(self._shared_values['manager_best_push_queue'], shared_best, self.host, self.port)
                next_iter_counter += 1

                # getting best from other players via manager
                res = evaluate_call(None, best_rpc)
                best_solutions = res[0]
                best_solutions[obj] = shared_best['solution']
                iters = res[1]
                iters[obj] = shared_population['iteration']
                nobjs = len(best_solutions)
                best_mask = np.zeros(nobjs, dtype=bool)
                iters_mask = np.zeros(nobjs, dtype=bool)

                for i in range(nobjs):
                    if best_solutions_pop is None or np.any(best_solutions_pop[i] != best_solutions[i]):
                        best_mask[i] = True
                    if iters_pop is None or iters_pop[i] != iters[i]:
                        iters_mask[i] = True

                # exchange gens
                if self._shared_values['next_iter'] <= 0 or self._shared_values['next_iter'] - next_iter_counter >= 0:
                    if np.any(best_mask[:obj]) or np.any(best_mask[obj + 1:]):
                        for i in range(pop.shape[0]):
                            for j in range(nobjs):
                                if j != obj and best_solutions[j] is not None:
                                    pop[i][patterns[j]] = best_solutions[j][patterns[j]]
                        if self.repair_rpc is not None:
                            pop = evaluate_call(pop, self.repair_rpc)
                        pop_eval = evaluate_call(pop, self.obj_rpc)
                        evaluation_counter += pop.shape[0]

                if np.all(iters_mask[:obj]) and np.all(iters_mask[obj + 1:]):
                    next_iter_counter = 0
                    iters_pop = deepcopy(iters)

                if self._shared_values['start_flag']: # and np.all(iters_mask):
                    best_solutions_pop = deepcopy(best_solutions)
                    unique_pop, indices = np.unique(pop, axis=0, return_index=True)
                    unique_pop_eval = pop_eval[indices]
                    shared_population['population'] = deepcopy(unique_pop)
                    shared_population['population_eval'] = deepcopy(unique_pop_eval)
                    shared_population['evaluation_counter'] = evaluation_counter
                    _send_to(self._shared_values['manager_pop_queue'], shared_population, self.host, self.port)
            else:
                first = True
                next_iter_counter = 0
                shared_best['solution'] = None
                shared_best['iter_counter'] = 0

    @staticmethod
    def _cmd_consumer(self):
        connection = pika.BlockingConnection(pika.ConnectionParameters(host=self.host, port=self.port))
        channel = connection.channel()
        channel.exchange_declare(exchange=self.cmd_exchange, exchange_type='fanout')
        result = channel.queue_declare(queue='', exclusive=True)
        queue_name = result.method.queue
        channel.queue_bind(exchange=self.cmd_exchange, queue=queue_name)

        def callback(cal_shared_values, cal_lock, body):
            msg = pickle.loads(body, encoding='bytes')
            if msg[0] == 'cmd':
                if msg[1] == 'Start':
                    with cal_lock:
                        cal_shared_values['start_flag'] = True
                elif msg[1] == 'Stop':
                    with cal_lock:
                        cal_shared_values['start_flag'] = False

            elif msg[0] == 'parm':
                if msg[1] == 'Gens':
                    with cal_lock:
                        cal_shared_values['patterns'] = msg[2]
                elif msg[1] == 'Next Iter':
                    with cal_lock:
                        cal_shared_values['next_iter'] = msg[2]

            elif msg[0] == 'queue':
                if msg[1] == 'Manager Best':
                    with cal_lock:
                        cal_shared_values['manager_best_push_queue'] = msg[2][0]
                        cal_shared_values['manager_best_pull_queue'] = msg[2][1]
                elif msg[1] == 'Manager Pop':
                    with cal_lock:
                        cal_shared_values['manager_pop_queue'] = msg[2]

        channel.basic_consume(queue=queue_name,
                              on_message_callback=lambda ch, met, prop, body: callback(self._shared_values, self._lock,
                                                                                       body),
                              auto_ack=True)
        channel.start_consuming()

    @staticmethod
    def _create_individual_uniform(bounds):
        a = np.array([bounds[k][0] for k in range(len(bounds))])
        b = np.array([bounds[k][1] for k in range(len(bounds))])
        return np.random.uniform(a, b)


def _send_to(queue, msg, host, port):
    message = pickle.dumps(msg)
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=host, port=port))
    channel = connection.channel()
    channel.queue_declare(queue=queue)
    channel.basic_publish(exchange='', routing_key=queue, body=message)
    connection.close()
