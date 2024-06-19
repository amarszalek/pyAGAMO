from osbrain import run_agent
from osbrain import run_nameserver


class Repair:
    def __init__(self, ns=None, transport='ipc', args=None, verbose=False):
        self._ns = ns
        self.args = args
        self.transport = transport
        self.verbose = verbose
    
    def call(self, x, *args):
        """Abstract method. Override this method by formula of repair function.

        Parameters
        ----------
        x : numpy.ndarray
            A 2-d numpy array of solutions.
        args: tuple, optional
            An extra arguments if you need.

        Returns
        -------
        y : numpy.ndarray
            A 2-d numpy array of solutions.
        """
        raise NotImplementedError('You must override this method in your class!')
        
    def run(self, ns=None):
        if ns is None:
            self.ns = run_nameserver()
        else:
            self.ns = ns

        self.repair = run_agent(f'Repair', self.ns.addr(), transport=self.transport)
        self.addr = self.repair.bind('REP', alias='repair', handler=lambda a, m: self.reply(a, m),
                                     transport=self.transport)

        return self.addr
        
    def reply(self, agent, message):
        y = self.call(message, self.args)
        if self.verbose:
            agent.log_info(f'{agent.name}: repair {message.shape}')
        return y
