from pygamoo.objectives import RE34F1, RE34F2, RE34F3
from pygamoo.utils import print_running
import json

with open('../rabbithost.json', 'r') as f:
    rmq = json.load(f)

HOST = rmq['host']   # "rabbitmq host"
PORT = rmq['port']   # 5672

if __name__ == '__main__':
    objectives = [RE34F1(0, 'q_f1', host=HOST, port=PORT),
                  RE34F2(1, 'q_f2', host=HOST, port=PORT),
                  RE34F3(2, 'q_f3', host=HOST, port=PORT),
                  RE34F1(3, 'q_f1', host=HOST, port=PORT),
                  RE34F2(4, 'q_f2', host=HOST, port=PORT),
                  RE34F3(5, 'q_f3', host=HOST, port=PORT)
                  ]

    for obj in objectives:
        obj.run()
        print(f'Objective consumer is running at pid:{obj._p.pid} with queue name {obj.qname}')

    while True:
        print_running()

