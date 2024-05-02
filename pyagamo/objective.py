from osbrain import run_agent
from osbrain import run_nameserver
import Pyro4


@Pyro4.expose
class Objective:
    def __init__(self, num, n_var, n_obj, bounds, obj, ns=None, transport='ipc', args=None, verbose=False):
        self._num = num
        self._n_obj = n_obj
        self._n_var = n_var
        self._bounds = bounds
        self._obj = obj
        self.ns = ns
        self.args = args
        self.transport = transport
        self.verbose = verbose
    
    def call(self, x, *args):
        """Abstract method. Override this method by formula of objective function.

        Parameters
        ----------
        x : numpy.ndarray
            A 2-d numpy array of solutions.
        args: tuple, optional
            An extra arguments if you need.

        Returns
        -------
        y : numpy.ndarray
            A 1-d numpy array of values of objective function for given x.
        """
        raise NotImplementedError('You must override this method in your class!')
        
    def run(self, ns=None):
        if ns is None:
            self.ns = run_nameserver()
        else:
            self.ns = ns
        self.objective = run_agent(f'Objective_{self.num}', self.ns.addr(), transport=self.transport)
        self.addr = self.objective.bind('REP', alias='evaluate', handler=lambda a, m: self.reply(a, m),
                                        transport=self.transport)
        return self.addr
        
    def reply(self, agent, message):
        y = self.call(message, self.args)
        if self.verbose:
            agent.log_info(f'{agent.name}: evaluate {message.shape}')
        return y
        
    @property
    def n_obj(self):            
        return self._n_obj
    
    @n_obj.setter
    def n_obj(self, value):     
        self._n_obj = value
        
    @property
    def n_var(self):            
        return self._n_var
    
    @n_var.setter
    def n_var(self, value):     
        self._n_var = value
    
    @property
    def bounds(self):            
        return self._bounds
    
    @bounds.setter
    def bounds(self, value):     
        self._bounds = value
    
    @property
    def num(self):            
        return self._num

    @num.setter
    def num(self, value):     
        self._num = value
        
    @property
    def obj(self):            
        return self._obj

    @obj.setter
    def obj(self, value):     
        self._obj = value


def publish_objective(*Objs):
    daemon = Pyro4.Daemon()
    for Obj in Objs:
        uri = daemon.register(Obj)
        print(uri)
    daemon.requestLoop()
