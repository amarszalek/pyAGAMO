import pika
import pickle
from multiprocessing import Process


class Objective:
    def __init__(self, num, qname, host, port):
        self.num = num
        self.qname = qname
        self.host = host
        self.port = port
        self._p = None

    def __call__(self, x):
        raise NotImplementedError('You must override this method in your class!')

    def run(self):
        p = Process(target=self._consumer, args=(self,))
        p.daemon = True
        p.start()
        self._p = p

    def close(self):
        if self._p is not None:
            self._p.terminate()
        connection = pika.BlockingConnection(pika.ConnectionParameters(host=self.host, port=self.port))
        channel = connection.channel()
        channel.queue_delete(queue=self.qname)

    def __del__(self):
        self.close()

    @staticmethod
    def _consumer(self):
        def on_request(ch, method, props, body):
            x = pickle.loads(body, encoding='bytes')
            response = pickle.dumps(self(x))

            ch.basic_publish(exchange='', routing_key=props.reply_to, properties=pika.BasicProperties(
                correlation_id=props.correlation_id), body=response)
            ch.basic_ack(delivery_tag=method.delivery_tag)
        connection = pika.BlockingConnection(pika.ConnectionParameters(host=self.host, port=self.port))
        channel = connection.channel()
        channel.queue_declare(queue=self.qname)
        channel.basic_qos(prefetch_count=1)
        channel.basic_consume(queue=self.qname, on_message_callback=on_request)
        channel.start_consuming()
