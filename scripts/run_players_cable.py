from pygamoo.players import ClonalSelection
from pygamoo.utils import print_running
import json

with open('../rabbithost_ser.json', 'r') as f:
    rmq = json.load(f)

HOST = rmq['host']   # "rabbitmq host"
PORT = rmq['port']   # 5672

# Cable problem
bounds = list(zip([0.2, 0.2, 0.2, 0.2, 1.0, 1.0, 0.0], [0.6, 0.6, 0.6, 0.6, 2.2, 1.6, 1.0]))
#bounds = list(zip([0.2, 0.2, 0.2, 0.2, 1.0, 1.2, 0.0], [0.5, 0.5, 0.5, 0.5, 2.2, 1.6, 1.0]))
pop_size = 20
nvars = 7
player_parm = {"nclone": 10, "mutate_args": [0.45, 0.9, 0.1]}


if __name__ == '__main__':
    players = [ClonalSelection(0, 'q_f1', None, 'pls_exchange', 25, nvars, bounds, HOST, PORT, player_parm),
               ClonalSelection(1, 'q_f2', None, 'pls_exchange', 25, nvars, bounds, HOST, PORT, player_parm),
               ClonalSelection(2, 'q_f3', None, 'pls_exchange', 10, nvars, bounds, HOST, PORT, player_parm)]

    for player in players:
        player.run()
        print(f'Player {player.num+1} main process and consumer are running at pids :{player._p1.pid}',
              f'and {player._p2.pid} with objective queue: {player.obj_queue}',
              f'and command exchange: {player.cmd_exchange}')
    while True:
        print_running()
