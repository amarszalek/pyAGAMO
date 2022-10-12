from pygamoo.objectives import CableF1, CableF2, CableF3
from pygamoo.utils import print_running
import json

with open('../rabbithost_ser.json', 'r') as f:
    rmq = json.load(f)

HOST = rmq['host']   # "rabbitmq host"
PORT = rmq['port']   # 5672

if __name__ == '__main__':
    objectives = [CableF1(0, 'q_f1', host=HOST, port=PORT),
                  CableF2(1, 'q_f2', host=HOST, port=PORT),
                  CableF3(2, 'q_f3', host=HOST, port=PORT)
                  ]

    for obj in objectives:
        obj.run()
        print(f'Objective consumer is running at pid:{obj._p.pid} with queue name {obj.queue}')

    while True:
        print_running()

