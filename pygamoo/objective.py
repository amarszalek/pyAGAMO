import pika
import pickle
from multiprocessing import Process
from datetime import datetime

try:
    import matlab.engine
except ModuleNotFoundError:
    pass


class Objective:
    """Base class for creating objective function objects.
    It provides functionalities for evaluate objective function via message broker.

    Parameters
    ----------
    num : int
        Unique number of Objective.
    queue : str
        The queue name. It is using by main algorithm and players to identify the objective function.
    host : str
        Hostname or IP Address to connect to message broker.
    port : int
        TCP port to connect to message broker.
    log_queue : str, optional
        The logger queue name.
    args : tuple, optional
        An extra arguments if you need.
    """
    def __init__(self, num, queue, host, port, log_queue=None, args=None):
        """Constructor method
        """
        self.num = num
        self.queue = queue
        self.host = host
        self.port = port
        self.log_queue = log_queue
        self._p = None
        self.args = args

    def call(self, x, *args):
        """Abstract method. Override this method by formula of objective function.

        Parameters
        ----------
        x : numpy.ndarray
            A 2-d numpy array of solutions.
        args: tuple, optional
            An extra arguments if you need.

        Returns
        -------
        y : numpy.ndarray
            A 1-d numpy array of values of objective function for given x.
        """
        raise NotImplementedError('You must override this method in your class!')

    def run(self):
        p = Process(target=self._consumer, args=(self,))
        p.daemon = True
        p.start()
        self._p = p
        if self.log_queue:
            self._send_to_logger(f'{datetime.now()}: Objective num: {self.num} was started')

    def is_alive(self):
        if self._p is not None:
            return self._p.is_alive()
        return False

    def close(self):
        if self._p is not None:
            self._p.terminate()
            if self.log_queue:
                self._send_to_logger(f'{datetime.now()}: Objective num: {self.num} was closed')
            connection = pika.BlockingConnection(pika.ConnectionParameters(host=self.host, port=self.port))
            channel = connection.channel()
            q_state = channel.queue_declare(self.queue)
            if q_state.method.consumer_count == 0:
                channel.queue_delete(queue=self.queue)

    def __del__(self):
        self.close()

    @staticmethod
    def _consumer(self):
        def on_request(ch, method, props, body):
            try:
                x = pickle.loads(body, encoding='bytes')
                res = self.call(x, self.args)
                if self.log_queue:
                    self._send_to_logger(f'{datetime.now()}: Objective num: {self.num} evaluated solutions,'
                                         f' shape: {x.shape}')
                response = pickle.dumps(res)
            except Exception as e:
                if self.log_queue:
                    self._send_to_logger(f'{datetime.now()}: Objective num: {self.num} exception: {type(e)}')
                response = pickle.dumps(e)

            ch.basic_publish(exchange='', routing_key=props.reply_to, properties=pika.BasicProperties(
                correlation_id=props.correlation_id), body=response)
            ch.basic_ack(delivery_tag=method.delivery_tag)

        connection = pika.BlockingConnection(pika.ConnectionParameters(host=self.host, port=self.port))
        channel = connection.channel()
        channel.queue_declare(queue=self.queue)
        channel.basic_qos(prefetch_count=1)
        channel.basic_consume(queue=self.queue, on_message_callback=on_request)
        channel.start_consuming()

    def _send_to_logger(self, msg):
        message = pickle.dumps(msg)
        connection = pika.BlockingConnection(pika.ConnectionParameters(host=self.host, port=self.port))
        channel = connection.channel()
        channel.queue_declare(queue=self.log_queue)
        channel.basic_publish(exchange='', routing_key=self.log_queue, body=message)
        connection.close()


class ObjectiveMatlabEngine(Objective):
    """Base class for creating objective function objects that use Matlab engine.
    """

    def call(self, x, *args):
        raise NotImplementedError('You must override this method in your class!')

    @staticmethod
    def _consumer(self):
        eng = matlab.engine.start_matlab()

        def on_request(ch, method, props, body):
            try:
                x = pickle.loads(body, encoding='bytes')
                res = self.call(x, eng, self.args)
                if self.log_queue:
                    self._send_to_logger(f'{datetime.now()}: Objective num: {self.num} evaluated solutions,'
                                         f' shape: {x.shape}')
                response = pickle.dumps(res)
            except Exception as e:
                if self.log_queue:
                    self._send_to_logger(f'{datetime.now()}: Objective num: {self.num} exception: {e}')
                response = pickle.dumps(e)

            ch.basic_publish(exchange='', routing_key=props.reply_to, properties=pika.BasicProperties(
                correlation_id=props.correlation_id), body=response)
            ch.basic_ack(delivery_tag=method.delivery_tag)

        connection = pika.BlockingConnection(pika.ConnectionParameters(host=self.host, port=self.port))
        channel = connection.channel()
        channel.queue_declare(queue=self.queue)
        channel.basic_qos(prefetch_count=1)
        channel.basic_consume(queue=self.queue, on_message_callback=on_request)
        channel.start_consuming()
