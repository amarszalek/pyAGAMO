import pika
import uuid
import pickle
import numpy as np


class RpcClient:
    def __init__(self, qname, host, port):
        self.queue_name = qname
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host=host, port=port))
        self.channel = self.connection.channel()
        result = self.channel.queue_declare(queue='', exclusive=True)
        self.callback_queue = result.method.queue
        self.channel.basic_consume(queue=self.callback_queue, on_message_callback=self.on_response, auto_ack=True)
        self.response = None
        self.corr_id = None

    def on_response(self, ch, method, props, body):
        if self.corr_id == props.correlation_id:
            self.response = body

    def call(self, n):
        self.response = None
        self.corr_id = str(uuid.uuid4())
        self.channel.basic_publish(exchange='', routing_key=self.queue_name,
                                   properties=pika.BasicProperties(reply_to=self.callback_queue,
                                                                   correlation_id=self.corr_id), body=n)
        while self.response is None:
            self.connection.process_data_events()
        return self.response


def evaluate_call(solutions, rpc):
    res = rpc.call(pickle.dumps(solutions))
    return pickle.loads(res, encoding='bytes')


def assigning_gens(nvars, nobjs):
    while True:
        if nvars <= nobjs:
            r = np.random.choice(range(nobjs), size=(nvars,), replace=False)
            r2 = np.stack([r == i for i in range(nobjs)])
            break
        else:
            r = np.random.randint(0, nobjs, size=(nvars,))
            r2 = np.stack([r == i for i in range(nobjs)])
            if nvars >= nobjs and not np.any(np.all(r2, axis=1)) and not np.any(np.all(np.logical_not(r2), axis=1)):
                break
    return r2


def pairwise_dominance(x):
    z = x[:, np.newaxis] >= x
    z = np.all(z, axis=2)
    z[range(z.shape[0]), range(z.shape[0])] = False
    xx = np.any(z, axis=1)
    return np.logical_not(xx)


def get_not_dominated(populations_eval):
    mask = pairwise_dominance(populations_eval)
    return mask


def pairwise_distance(x):
    return np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)


def pairwise_distance_xy(x, y):
    return np.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1)


def front_suppression(front_eval, front_max):
    n = front_eval.shape[0] - front_max
    z = pairwise_distance(front_eval)
    ran = np.random.random()
    if ran > 0.5:
        z = np.transpose(z)
    t = np.tril(z) + np.triu(np.ones_like(z) * 1000000)
    arg = np.argsort(t, axis=None)
    indx = np.unravel_index(arg, t.shape)[0]
    u, i = np.unique(indx, return_index=True)
    mask = np.ones(front_eval.shape[0], dtype=bool)
    mask[indx[np.sort(i)][:n]] = False
    return mask
