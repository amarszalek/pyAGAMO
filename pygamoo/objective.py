import pika
import pickle
from multiprocessing import Process
from datetime import datetime


class Objective:
    """Base class for creating objective function objects.
    It provides functionalities for evaluate objective function via message broker.

    Parameters
    ----------
    num : int
        Unique number of Objective
    qname : str
        The queue name. It is using by main algorithm and players to identify the objective function.
    host : str
        Hostname or IP Address to connect to message broker
    port : int
        TCP port to connect to message broker
    """
    def __init__(self, num, qname, host, port, log_queue=None):
        """Constructor method
        """
        self.num = num
        self.qname = qname
        self.host = host
        self.port = port
        self.log_queue = log_queue
        self._p = None

    def call(self, x):
        """Abstract method. Override this method by formula of objective function.

        Parameters
        ----------
        x : numpy.ndarray
            A 2-d numpy array of solutions.

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
        if self.log_queue is not None:
            self._send_to(self.log_queue, f'{datetime.now()}: Objective num: {self.num} was started', self.host,
                          self.port)

    def is_alive(self):
        if self._p is not None:
            return self._p.is_alive()
        return False

    def close(self):
        if self._p is not None:
            self._p.terminate()
            if self.log_queue is not None:
                self._send_to(self.log_queue, f'{datetime.now()}: Objective num: {self.num} was closed', self.host,
                              self.port)
            #connection = pika.BlockingConnection(pika.ConnectionParameters(host=self.host, port=self.port))
            #channel = connection.channel()
            #channel.queue_delete(queue=self.qname)

    def __del__(self):
        self.close()

    @staticmethod
    def _consumer(self):
        def on_request(ch, method, props, body):
            try:
                x = pickle.loads(body, encoding='bytes')
                res = self.call(x)
                response = pickle.dumps(res)
            except Exception as e:
                response = pickle.dumps(e)

            ch.basic_publish(exchange='', routing_key=props.reply_to, properties=pika.BasicProperties(
                correlation_id=props.correlation_id), body=response)
            ch.basic_ack(delivery_tag=method.delivery_tag)
        connection = pika.BlockingConnection(pika.ConnectionParameters(host=self.host, port=self.port))
        channel = connection.channel()
        channel.queue_declare(queue=self.qname)
        channel.basic_qos(prefetch_count=1)
        channel.basic_consume(queue=self.qname, on_message_callback=on_request)
        channel.start_consuming()

    @staticmethod
    def _send_to(queue, msg, host, port):
        message = pickle.dumps(msg)
        connection = pika.BlockingConnection(pika.ConnectionParameters(host=host, port=port))
        channel = connection.channel()
        channel.queue_declare(queue=queue)
        channel.basic_publish(exchange='', routing_key=queue, body=message)
        connection.close()
