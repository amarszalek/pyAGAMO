import numpy as np
from pyagamo import Objective


class DTLZ(Objective):
    def __init__(self, num, n_dtlz=1, obj=1, n_var=10, n_obj=3, alpha=100, d=100, ns=None, transport='ipc', args=None,
                 verbose=False):
        self.n_dtlz = n_dtlz
        obj = obj - 1
        n_var = n_var
        n_obj = n_obj
        self.k = n_var - n_obj + 1
        self.alpha = alpha
        self.d = d
        bounds = list(zip([0.0]*self.n_var, [1.0]*self.n_var))
        super(DTLZ, self).__init__(num, n_var, n_obj, bounds, obj, ns, transport, args, verbose)

    def g1(self, X_M):
        return 100 * (self.k + np.sum(np.square(X_M - 0.5) - np.cos(20 * np.pi * (X_M - 0.5)), axis=1))

    def g2(self, X_M):
        return np.sum(np.square(X_M - 0.5), axis=1)
    
    def obj_func(self, X_, g, alpha=1):
        for i in range(0, self.n_obj):
            _f = (1 + g)
            _f *= np.prod(np.cos(np.power(X_[:, :X_.shape[1] - i], alpha) * np.pi / 2.0), axis=1)
            if i > 0:
                _f *= np.sin(np.power(X_[:, X_.shape[1] - i], alpha) * np.pi / 2.0)
            if i == self.obj:
                return _f
            
    def obj_func1(self, X_, g):
        for i in range(0, self.n_obj):
            _f = 0.5 * (1 + g)
            _f *= np.prod(X_[:, :X_.shape[1] - i], axis=1)
            if i > 0:
                _f *= 1 - X_[:, X_.shape[1] - i]
            if i == self.obj:
                return _f

    def call(self, x, *args):
        if self.n_dtlz == 1:
            X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
            g = self.g1(X_M)
            return self.obj_func1(X_, g)
        elif self.n_dtlz == 2:
            X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
            g = self.g2(X_M)
            return self.obj_func(X_, g, alpha=1)
        elif self.n_dtlz == 3:
            X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
            g = self.g1(X_M)
            return self.obj_func(X_, g, alpha=1)
        elif self.n_dtlz == 4:
            X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
            g = self.g2(X_M)
            return self.obj_func(X_, g, alpha=self.alpha)
        elif self.n_dtlz == 5:
            X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
            g = self.g2(X_M)
            theta = 1 / (2 * (1 + g[:, None])) * (1 + 2 * g[:, None] * X_)
            theta = np.column_stack([x[:, 0], theta[:, 1:]])
            return self.obj_func(theta, g)
        elif self.n_dtlz == 6:
            X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
            g = np.sum(np.power(X_M, 0.1), axis=1)
            theta = 1 / (2 * (1 + g[:, None])) * (1 + 2 * g[:, None] * X_)
            theta = np.column_stack([x[:, 0], theta[:, 1:]])
            return self.obj_func(theta, g)
        elif self.n_dtlz == 7:
            if self.obj < self.n_obj - 1:
                return x[:, self.obj]
            else:
                f = []
                for i in range(0, self.n_obj - 1):
                    f.append(x[:, i])
                f = np.column_stack(f)
                g = 1 + 9 / self.k * np.sum(x[:, -self.k:], axis=1)
                h = self.n_obj - np.sum(f / (1 + g[:, None]) * (1 + np.sin(3 * np.pi * f)), axis=1)
                return (1+g)*h