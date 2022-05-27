import pika
import pickle
from multiprocessing import Process


class Objective:
    """Base class for creating objective function objects.
    It provides functionalities for evaluate objective function via message broker.

    Parameters
    ----------
    qname : str
        The queue name. It is using by main algorithm and players to identify the objective function.
    host : str
        Hostname or IP Address to connect to message broker
    port : int
        TCP port to connect to message broker
    """
    def __init__(self, qname, host, port):
        """Constructor method
        """
        self.qname = qname
        self.host = host
        self.port = port
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

    def is_alive(self):
        if self._p is not None:
            return self._p.is_alive()
        return False

    def close(self):
        if self._p is not None:
            self._p.terminate()
            #connection = pika.BlockingConnection(pika.ConnectionParameters(host=self.host, port=self.port))
            #channel = connection.channel()
            #channel.queue_delete(queue=self.qname)

    def __del__(self):
        self.close()

    @staticmethod
    def _consumer(self):
        def on_request(ch, method, props, body):
            x = pickle.loads(body, encoding='bytes')
            response = pickle.dumps(self.call(x))

            ch.basic_publish(exchange='', routing_key=props.reply_to, properties=pika.BasicProperties(
                correlation_id=props.correlation_id), body=response)
            ch.basic_ack(delivery_tag=method.delivery_tag)
        connection = pika.BlockingConnection(pika.ConnectionParameters(host=self.host, port=self.port))
        channel = connection.channel()
        channel.queue_declare(queue=self.qname)
        channel.basic_qos(prefetch_count=1)
        channel.basic_consume(queue=self.qname, on_message_callback=on_request)
        channel.start_consuming()
