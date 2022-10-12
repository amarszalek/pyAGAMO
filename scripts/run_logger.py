from pygamoo.logger import Logger
from pygamoo.utils import print_running
import json

with open('../rabbithost.json', 'r') as f:
    rmq = json.load(f)

HOST = rmq['host']   # "rabbitmq host"
PORT = rmq['port']   # 5672

if __name__ == '__main__':
    logger = Logger(r'../temp/log.txt', 'logs', host=HOST, port=PORT)
    logger.run()
    print(f'Logger is running at pid:{logger._p.pid} with queue name {logger.queue}')

    while True:
        print_running()
