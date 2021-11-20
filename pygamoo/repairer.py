from pygamoo.objective import Objective


class Repairer(Objective):
    def __call__(self, x):
        raise NotImplementedError('You must override this method in your class!')
