import numpy as np
import pika
import pickle
import time
import random
from copy import deepcopy
from multiprocessing import Process, Manager
from pygamoo.utils import RpcClient, assigning_gens
from pygamoo.utils import evaluate_call, get_not_dominated, front_suppression


class AGAMOO:
    def __init__(self, nobjs, nvars, max_eval, change_iter, next_iter, max_front, cmd_exchange, objs_queues, host,
                 port):
        self.nobjs = nobjs
        self.nvars = nvars
        self.max_front = max_front
        self.max_eval = max_eval
        self.change_iter = change_iter
        self.next_iter = next_iter
        self._prefix = ''.join([random.choice('abcdefghijklmnoprstquwyxz') for _ in range(10)])
        self.best_pull_queue = self._prefix + '_best_pull'
        self.best_push_queue = self._prefix + '_best_push'
        self.pop_queue = self._prefix + '_pop'
        self.cmd_queue = self._prefix + '_cmd'
        self.cmd_exchange = cmd_exchange
        self.objs_queues = objs_queues
        self.host = host
        self.port = port
        self._p1 = None
        self._p2 = None
        self._p3 = None
        self._p4 = None
        self._p5 = None
        try:
            manager = getattr(type(self), 'manager')
        except AttributeError:
            manager = type(self).manager = Manager()
        self._shared_best = manager.dict({})
        self._shared_best['solutions'] = [None] * self.nobjs
        self._shared_best['iter_counters'] = [None] * self.nobjs
        self._shared_front = manager.dict({})
        self._shared_front['objs_rpc'] = self.objs_queues
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
        self._shared_values = manager.dict({})
        self._shared_values['start_flag'] = False
        self._lock = manager.RLock()

    def get_results(self):
        res = deepcopy(self._shared_front)
        del res['objs_rpc']
        del res['nobjs']
        del res['max_front']
        del res['change_iter']
        del res['max_eval']
        del res['min_iter_pop']
        del res['change_flag']
        return res

    def start(self):
        self._send_to(self.cmd_queue, 'Start')

    def stop(self):
        self._send_to(self.cmd_queue, 'Stop')

    def close(self):
        msg = ['cmd', 'Stop']
        self._send_to_players(pickle.dumps(msg))
        time.sleep(1)
        if self._p1 is not None:
            self._p1.terminate()
            self._p1 = None
        if self._p2 is not None:
            self._p2.terminate()
            self._p2 = None
        if self._p3 is not None:
            self._p3.terminate()
            self._p3 = None
        if self._p4 is not None:
            self._p4.terminate()
            self._p4 = None
        if self._p5 is not None:
            self._p5.terminate()
            self._p5 = None
        connection = pika.BlockingConnection(pika.ConnectionParameters(host=self.host, port=self.port))
        channel = connection.channel()
        channel.queue_delete(queue=self.best_pull_queue)
        channel.queue_delete(queue=self.best_push_queue)
        channel.queue_delete(queue=self.pop_queue)
        channel.queue_delete(queue=self.cmd_queue)

    def __del__(self):
        self.close()

    def run(self):
        p1 = Process(target=self._best_pull_consumer, args=(self,))
        p1.daemon = True
        p1.start()

        p2 = Process(target=self._best_push_consumer, args=(self,))
        p2.daemon = True
        p2.start()

        p3 = Process(target=self._pop_consumer, args=(self,))
        p3.daemon = True
        p3.start()

        p4 = Process(target=self._cmd_consumer, args=(self,))
        p4.daemon = True
        p4.start()

        p5 = Process(target=self._main_process, args=(self,))
        p5.daemon = True
        p5.start()

        self._p1 = p1
        self._p2 = p2
        self._p3 = p3
        self._p4 = p4
        self._p5 = p5

    def is_alive(self, separate=False):
        if separate:
            return (False if self._p1 is None else self._p1.is_alive(),
                    False if self._p2 is None else self._p2.is_alive(),
                    False if self._p3 is None else self._p3.is_alive(),
                    False if self._p4 is None else self._p4.is_alive(),
                    False if self._p5 is None else self._p5.is_alive())

        if (self._p1 is not None) and (self._p2 is not None) and (self._p3 is not None) and (self._p4 is not None) and\
                (self._p5 is not None):
            return all([self._p1.is_alive(), self._p2.is_alive(), self._p3.is_alive(), self._p4.is_alive(),
                        self._p5.is_alive()])
        return False

    def is_working(self):
        return self._shared_values['start_flag']

    @staticmethod
    def _best_pull_consumer(self,):
        connection = pika.BlockingConnection(pika.ConnectionParameters(host=self.host, port=self.port))
        channel = connection.channel()
        channel.queue_declare(queue=self.best_pull_queue)

        def callback(cal_shared_best, ch, method, props):
            best = [deepcopy(cal_shared_best['solutions']), deepcopy(cal_shared_best['iter_counters'])]
            response = pickle.dumps(best)
            ch.basic_publish(exchange='', routing_key=props.reply_to,
                             properties=pika.BasicProperties(correlation_id=props.correlation_id), body=response)
            ch.basic_ack(delivery_tag=method.delivery_tag)

        channel.basic_qos(prefetch_count=1)
        channel.basic_consume(queue=self.best_pull_queue,
                              on_message_callback=lambda ch, met, prop, body: callback(self._shared_best, ch,
                                                                                       met, prop))
        channel.start_consuming()

    @staticmethod
    def _best_push_consumer(self,):
        connection = pika.BlockingConnection(pika.ConnectionParameters(host=self.host, port=self.port))
        channel = connection.channel()
        channel.queue_declare(queue=self.best_push_queue)

        def callback(cal_shared_best, cal_lock, body):
            player_data = pickle.loads(body, encoding='bytes')
            nobj = player_data['nobj']
            solution = player_data['solution']
            iterr = player_data['iter_counter']
            solutions = deepcopy(cal_shared_best['solutions'])
            iterrs = deepcopy(cal_shared_best['iter_counters'])
            solutions[nobj] = solution
            iterrs[nobj] = iterr
            with cal_lock:
                cal_shared_best['solutions'] = solutions
                cal_shared_best['iter_counters'] = iterrs

        channel.basic_consume(queue=self.best_push_queue,
                              on_message_callback=lambda ch, met, prop, body: callback(self._shared_best, self._lock,
                                                                                       body),
                              auto_ack=True)
        channel.start_consuming()

    @staticmethod
    def _pop_consumer(self,):
        connection = pika.BlockingConnection(pika.ConnectionParameters(host=self.host, port=self.port))
        channel = connection.channel()
        channel.queue_declare(queue=self.pop_queue)

        def callback(cal_shared_front, cal_lock, body):
            front = cal_shared_front['front']
            front_eval = cal_shared_front['front_eval']
            evaluations = cal_shared_front['evaluations']
            evaluations_m = cal_shared_front['evaluations_m']
            iterations = cal_shared_front['iterations']
            max_front = cal_shared_front['max_front']
            change_flag = cal_shared_front['change_flag']
            change_iter = cal_shared_front['change_iter']
            min_iter_pop = cal_shared_front['min_iter_pop']
            stop_flag = cal_shared_front['stop_flag']
            max_eval = cal_shared_front['max_eval']
            nobjs = cal_shared_front['nobjs']

            if stop_flag:
                return None

            objs_rpc = []
            for i in range(nobjs):
                objs_rpc.append(RpcClient(cal_shared_front['objs_rpc'][i], self.host, self.port))

            player_data = pickle.loads(body, encoding='bytes')
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

            if len(evaluations) == 0:
                evaluations = np.zeros(nobjs)
            evaluations[nobj] = evaluation_counter

            pop_evals = []
            if len(evaluations_m) == 0:
                evaluations_m = np.zeros(nobjs)

            if pop.shape[0] > 0:
                for i in range(nobjs):
                    if i != nobj:
                        pop_evals.append(np.reshape(evaluate_call(pop, objs_rpc[i]), (-1, 1)))
                        evaluations_m[i] += pop.shape[0]
                    else:
                        pop_evals.append(np.reshape(pop_eval, (-1, 1)))
                pop_evals = np.hstack(pop_evals)

            if max_eval > 0:
                if np.min(evaluations + evaluations_m) >= max_eval:
                    stop_flag = True
                    msg = ['cmd', 'Stop']
                    self._send_to_players(pickle.dumps(msg))
            if pop.shape[0] > 0:
                mask = get_not_dominated(pop_evals)
                pop = pop[mask]
                pop_evals = pop_evals[mask]

                if len(front) == 0:
                    front = deepcopy(pop)
                    front_eval = deepcopy(pop_evals)
                else:
                    front = np.vstack([front, pop])
                    front_eval = np.vstack([front_eval, pop_evals])
                    mask = get_not_dominated(front_eval)
                    front = front[mask]
                    front_eval = front_eval[mask]

            # front suppression
            if 0 < max_front < front.shape[0]:
                mask = front_suppression(front_eval, max_front)
                front = front[mask]
                front_eval = front_eval[mask]

            with cal_lock:
                cal_shared_front['front'] = front
                cal_shared_front['front_eval'] = front_eval
                cal_shared_front['evaluations'] = evaluations
                cal_shared_front['evaluations_m'] = evaluations_m
                cal_shared_front['iterations'] = iterations
                cal_shared_front['change_flag'] = change_flag
                cal_shared_front['stop_flag'] = stop_flag
                cal_shared_front['min_iter_pop'] = min_iter_pop

        channel.basic_consume(queue=self.pop_queue,
                              on_message_callback=lambda ch, met, prop, body: callback(self._shared_front, self._lock,
                                                                                       body),
                              auto_ack=True)
        channel.start_consuming()

    def _send_to_players(self, message):
        connection = pika.BlockingConnection(pika.ConnectionParameters(host=self.host, port=self.port))
        channel = connection.channel()
        channel.exchange_declare(exchange=self.cmd_exchange, exchange_type='fanout')
        channel.basic_publish(exchange=self.cmd_exchange, routing_key='', body=message)
        connection.close()

    def _send_to(self, queue, msg):
        message = pickle.dumps(msg)
        connection = pika.BlockingConnection(pika.ConnectionParameters(host=self.host, port=self.port))
        channel = connection.channel()
        channel.queue_declare(queue=queue)
        channel.basic_publish(exchange='', routing_key=queue, body=message)
        connection.close()

    @staticmethod
    def _cmd_consumer(self):
        connection = pika.BlockingConnection(pika.ConnectionParameters(host=self.host, port=self.port))
        channel = connection.channel()
        channel.queue_declare(queue=self.cmd_queue)

        def callback(cal_shared_values, cal_shared_front, cal_lock, body):
            cmd = pickle.loads(body, encoding='bytes')
            with cal_lock:
                if cmd == 'Start':
                    cal_shared_values['start_flag'] = True
                    cal_shared_front['stop_flag'] = False
                if cmd == 'Stop':
                    cal_shared_values['start_flag'] = False

        channel.basic_consume(queue=self.cmd_queue,
                              on_message_callback=lambda ch, met, prop, body: callback(self._shared_values,
                                                                                       self._shared_front, self._lock,
                                                                                       body),
                              auto_ack=True)
        channel.start_consuming()

    @staticmethod
    def _main_process(self):
        first = True
        while True:
            if self._shared_front['stop_flag']:
                with self._lock:
                    self._shared_values['start_flag'] = not self._shared_front['stop_flag']
            if self._shared_values['start_flag']:
                if first:
                    with self._lock:
                        self._shared_front['objs_rpc'] = self.objs_queues
                        self._shared_front['front'] = []
                        self._shared_front['front_eval'] = []
                        self._shared_front['evaluations'] = []
                        self._shared_front['evaluations_m'] = []
                        self._shared_front['iterations'] = []
                        self._shared_front['min_iter_pop'] = 0
                        self._shared_front['change_flag'] = True
                        self._shared_front['stop_flag'] = False
                    # sending parms to players
                    msg = ['parm', 'Next Iter', self.next_iter]
                    self._send_to_players(pickle.dumps(msg))
                    msg = ['queue', 'Manager Best', [self.best_push_queue, self.best_pull_queue]]
                    self._send_to_players(pickle.dumps(msg))
                    msg = ['queue', 'Manager Pop', self.pop_queue]
                    self._send_to_players(pickle.dumps(msg))
                    first = False
                    # sending gens
                    if self._shared_front['change_flag']:
                        patterns = assigning_gens(self.nvars, self.nobjs)
                        msg = ['parm', 'Gens', patterns]
                        self._send_to_players(pickle.dumps(msg))
                        with self._lock:
                            self._shared_front['change_flag'] = False
                    # sending START to players
                    msg = ['cmd', 'Start']
                    self._send_to_players(pickle.dumps(msg))
                    continue
                # sending gens
                if self._shared_front['change_flag']:
                    patterns = assigning_gens(self.nvars, self.nobjs)
                    msg = ['parm', 'Gens', patterns]
                    self._send_to_players(pickle.dumps(msg))
                    with self._lock:
                        self._shared_front['change_flag'] = False
            else:
                if first:
                    continue
                else:
                    # sending STOP to players
                    msg = ['cmd', 'Stop']
                    self._send_to_players(pickle.dumps(msg))
                    # cleaning queues
                    connection = pika.BlockingConnection(pika.ConnectionParameters(host=self.host, port=self.port))
                    channel = connection.channel()
                    channel.queue_purge(queue=self.best_pull_queue)
                    channel.queue_purge(queue=self.best_push_queue)
                    channel.queue_purge(queue=self.pop_queue)
                    connection.close()
                    first = True
                    time.sleep(2)
