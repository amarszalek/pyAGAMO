from multiprocessing import Process
import pika
import pickle


class Logger:
    def __init__(self, log_file, queue, host, port):
        self.log_file = log_file
        self.queue = queue
        self.host = host
        self.port = port
        self._p = None
        # create logfile or clean if exist
        with open(self.log_file, 'w'):
            pass

    def run(self):
        p = Process(target=self._consumer, args=(self,))
        p.daemon = True
        p.start()
        self._p = p

    def close(self):
        if self._p is not None:
            self._p.terminate()
            self._p = None
        connection = pika.BlockingConnection(pika.ConnectionParameters(host=self.host, port=self.port))
        channel = connection.channel()
        q_state = channel.queue_declare(self.queue)
        if q_state.method.consumer_count == 0:
            channel.queue_delete(queue=self.queue)

    def __del__(self):
        self.close()

    def is_alive(self):
        return False if self._p is None else self._p.is_alive()

    @staticmethod
    def _consumer(self,):
        connection = pika.BlockingConnection(pika.ConnectionParameters(host=self.host, port=self.port))
        channel = connection.channel()
        channel.queue_declare(queue=self.queue)

        def callback(body):
            msg = pickle.loads(body, encoding='bytes')
            with open(self.log_file, 'a') as f:
                f.write(msg + '\n')

        channel.basic_consume(queue=self.queue,
                              on_message_callback=lambda ch, met, prop, body: callback(body), auto_ack=True)
        channel.start_consuming()
