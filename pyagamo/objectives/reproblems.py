import numpy as np
from pyagamo import Objective
import time


class RE21(Objective):
    def __init__(self, num, ns=None, obj=1, transport='ipc', args=None, verbose=False):
        obj = obj - 1
        n_var = 4
        n_obj = 2
        bounds = list(zip([1.0, np.sqrt(2.0), np.sqrt(2.0), 1.0], [3.0, 3.0, 3.0, 3.0]))
        super(RE21, self).__init__(num, n_var, n_obj, bounds, obj, ns, transport, args, verbose)
        
    def call(self, x, *args):
        F = 10.0
        sigma = 10.0
        E = 2.0 * 1e5
        L = 200.0
    
        if self.obj == 0:
            return L * ((2 * x[:,0]) + np.sqrt(2.0) * x[:,1] + np.sqrt(x[:,2]) + x[:,3])
        elif self.obj == 1:
            #time.sleep(0.000466*1.4)
            return ((F * L) / E) * ((2.0 / x[:,0]) + (2.0 * np.sqrt(2.0) / x[:,1]) - (2.0 * np.sqrt(2.0) / x[:,2]) + (2.0 / x[:,3]))
        else:
            raise ValueError('obj')


class RE22(Objective):
    def __init__(self, num, ns=None, obj=1, transport='ipc', args=None, verbose=False):
        obj = obj - 1
        n_var = 3
        n_obj = 2
        self.feasible_vals = np.array([0.20, 0.31, 0.40, 0.44, 0.60, 0.62, 0.79, 0.80, 0.88, 0.93, 1.0, 1.20, 1.24, 1.32, 1.40, 1.55, 1.58, 1.60, 1.76, 1.80, 1.86, 2.0, 2.17, 2.20, 2.37, 2.40, 2.48, 2.60, 2.64, 2.79, 2.80, 3.0, 3.08, 3,10, 3.16, 3.41, 3.52, 3.60, 3.72, 3.95, 3.96, 4.0, 4.03, 4.20, 4.34, 4.40, 4.65, 4.74, 4.80, 4.84, 5.0, 5.28, 5.40, 5.53, 5.72, 6.0, 6.16, 6.32, 6.60, 7.11, 7.20, 7.80, 7.90, 8.0, 8.40, 8.69, 9.0, 9.48, 10.27, 11.0, 11.06, 11.85, 12.0, 13.0, 14.0, 15.0])
        bounds = list(zip([0.2, 0.000001, 0.0], [15.0, 20.0, 40.0]))
        super(RE22, self).__init__(num, n_var, n_obj, bounds, obj, ns, transport, args, verbose)
        
    def call(self, x, *args):
        idx = np.abs(np.asarray(self.feasible_vals)[None, :] - x[:, 0:1]).argmin(axis=1)
        x1 = self.feasible_vals[idx]
        x2 = x[:, 1]
        x3 = x[:, 2]
    
        if self.obj == 0:
            return (29.4 * x1) + (0.6 * x2 * x3)
        elif self.obj == 1:
            g0 = (x1 * x3) - 7.735 * ((x1 * x1) / x2) - 180.0
            g1 = 4.0 - (x3 / x2)
            g0 = np.where(g0 < 0, -g0, 0)
            g1 = np.where(g1 < 0, -g1, 0)
            return g0 + g1
        else:
            raise ValueError('obj')

            
class RE23(Objective):
    def __init__(self, num, ns=None, obj=1, transport='ipc', args=None, verbose=False):
        obj = obj - 1
        n_var = 4
        n_obj = 2
        bounds = list(zip([1.0, 1.0, 10.0, 10.0], [100.0, 100.0, 200.0, 240.0]))
        super(RE23, self).__init__(num, n_var, n_obj, bounds, obj, ns, transport, args, verbose)
        
    def call(self, x, *args):
        x1 = 0.0625 * np.round(x[:, 0]).astype(int)
        x2 = 0.0625 * np.round(x[:, 1]).astype(int)
        x3 = x[:, 2]
        x4 = x[:, 3]

        # First original objective function
        if self.obj == 0:
            return (0.6224 * x1 * x3 * x4) + (1.7781 * x2 * x3 * x3) + (3.1661 * x1 * x1 * x4) + (19.84 * x1 * x1 * x3)
        elif self.obj == 1:
            # Original constraint functions
            g0 = x1 - (0.0193 * x3)
            g1 = x2 - (0.00954 * x3)
            g2 = (np.pi * x3 * x3 * x4) + ((4.0 / 3.0) * (np.pi * x3 * x3 * x3)) - 1296000
            g0 = np.where(g0 < 0, -g0, 0)
            g1 = np.where(g1 < 0, -g1, 0)
            g2 = np.where(g2 < 0, -g2, 0)
            return g0 + g1 + g2
        else:
            raise ValueError('obj')


class RE24(Objective):
    def __init__(self, num, ns=None, obj=1, transport='ipc', args=None, verbose=False):
        obj = obj - 1
        n_var = 2
        n_obj = 2
        bounds = list(zip([0.5, 0.5], [4, 50]))
        super(RE24, self).__init__(num, n_var, n_obj, bounds, obj, ns, transport, args, verbose)
        
    def call(self, x, *args):
        x1 = x[:, 0]
        x2 = x[:, 1]

        # First original objective function
        if self.obj == 0:
            return x1 + (120 * x2)
        elif self.obj == 1:
            E = 700000
            sigma_b_max = 700
            tau_max = 450
            delta_max = 1.5
            sigma_k = (E * x1 * x1) / 100
            sigma_b = 4500 / (x1 * x2)
            tau = 1800 / x2
            delta = (56.2 * 10000) / (E * x1 * x2 * x2)

            g0 = 1 - (sigma_b / sigma_b_max)
            g1 = 1 - (tau / tau_max)
            g2 = 1 - (delta / delta_max)
            g3 = 1 - (sigma_b / sigma_k)
            g0 = np.where(g0 < 0, -g0, 0)
            g1 = np.where(g1 < 0, -g1, 0)
            g2 = np.where(g2 < 0, -g2, 0)
            g3 = np.where(g3 < 0, -g3, 0)

            return g0 + g1 + g2 + g3
        else:
            raise ValueError('obj')
            

class RE25(Objective):
    def __init__(self, num, ns=None, obj=1, transport='ipc', args=None, verbose=False):
        obj = obj - 1
        n_var = 3
        n_obj = 2
        self.feasible_vals = np.array(
            [0.009, 0.0095, 0.0104, 0.0118, 0.0128, 0.0132, 0.014, 0.015, 0.0162, 0.0173, 0.018,
             0.02, 0.023, 0.025, 0.028, 0.032, 0.035, 0.041, 0.047, 0.054, 0.063, 0.072, 0.08,
             0.092, 0.105, 0.12, 0.135, 0.148, 0.162, 0.177, 0.192, 0.207, 0.225, 0.244, 0.263,
             0.283, 0.307, 0.331, 0.362, 0.394, 0.4375, 0.5])
        bounds = list(zip([1.0, 0.6, 0.09], [70, 30, 0.5]))
        super(RE25, self).__init__(num, n_var, n_obj, bounds, obj, ns, transport, args, verbose)
        
    def call(self, x, *args):
        x1 = np.round(x[:, 0])
        x2 = x[:, 1]
        idx = np.abs(np.asarray(self.feasible_vals)[None, :] - x[:, 2:3]).argmin(axis=1)
        x3 = self.feasible_vals[idx]

        # first original objective function
        if self.obj == 0:
            return (np.pi * np.pi * x2 * x3 * x3 * (x1 + 2)) / 4.0
        elif self.obj == 1:
            # constraint functions
            Cf = ((4.0 * (x2 / x3) - 1) / (4.0 * (x2 / x3) - 4)) + (0.615 * x3 / x2)
            Fmax = 1000.0
            S = 189000.0
            G = 11.5 * 1e+6
            K = (G * x3 * x3 * x3 * x3) / (8 * x1 * x2 * x2 * x2)
            lmax = 14.0
            lf = (Fmax / K) + 1.05 * (x1 + 2) * x3
            dmin = 0.2
            Dmax = 3
            Fp = 300.0
            sigmaP = Fp / K
            sigmaPM = 6
            sigmaW = 1.25

            g0 = -((8 * Cf * Fmax * x2) / (np.pi * x3 * x3 * x3)) + S
            g1 = -lf + lmax
            g2 = -3 + (x2 / x3)
            g3 = -sigmaP + sigmaPM
            g4 = -sigmaP - ((Fmax - Fp) / K) - 1.05 * (x1 + 2) * x3 + lf
            g5 = sigmaW - ((Fmax - Fp) / K)

            g0 = np.where(g0 < 0, -g0, 0)
            g1 = np.where(g1 < 0, -g1, 0)
            g2 = np.where(g2 < 0, -g2, 0)
            g3 = np.where(g3 < 0, -g3, 0)
            g4 = np.where(g4 < 0, -g4, 0)
            g5 = np.where(g5 < 0, -g5, 0)

            return g0 + g1 + g2 + g3 + g4 + g5
        else:
            raise ValueError('obj')


class RE31(Objective):
    def __init__(self, num, ns=None, obj=1, transport='ipc', args=None, verbose=False):
        obj = obj - 1
        n_var = 3
        n_obj = 3
        bounds = list(zip([0.00001, 0.00001, 1.0], [100.0, 100.0, 3.0]))
        super(RE31, self).__init__(num, n_var, n_obj, bounds, obj, ns, transport, args, verbose)
        
    def call(self, x, *args):
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]

        # First original objective function
        if self.obj == 0:
            return x1 * np.sqrt(16.0 + (x3 * x3)) + x2 * np.sqrt(1.0 + x3 * x3)
        elif self.obj == 1:
            # Second original objective function
            return (20.0 * np.sqrt(16.0 + (x3 * x3))) / (x1 * x3)
        elif self.obj == 2:
            # Constraint functions
            f0 = x1 * np.sqrt(16.0 + (x3 * x3)) + x2 * np.sqrt(1.0 + x3 * x3)
            f1 = (20.0 * np.sqrt(16.0 + (x3 * x3))) / (x1 * x3)
            g0 = 0.1 - f0
            g1 = 100000.0 - f1
            g2 = 100000 - ((80.0 * np.sqrt(1.0 + x3 * x3)) / (x3 * x2))
            g0 = np.where(g0 < 0, -g0, 0)
            g1 = np.where(g1 < 0, -g1, 0)
            g2 = np.where(g2 < 0, -g2, 0)
            return g0 + g1 + g2
        else:
            raise ValueError('obj')
            
            
class RE32(Objective):
    def __init__(self, num, ns=None, obj=1, transport='ipc', args=None, verbose=False):
        obj = obj - 1
        n_var = 4
        n_obj = 3
        bounds = list(zip([0.125, 0.1, 0.1, 0.125], [5.0, 10.0, 10.0, 5.0]))
        super(RE32, self).__init__(num, n_var, n_obj, bounds, obj, ns, transport, args, verbose)
        
    def call(self, x, *args):
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x4 = x[:, 3]

        P = 6000
        L = 14
        E = 30 * 1e6

        # // deltaMax = 0.25
        G = 12 * 1e6
        tauMax = 13600
        sigmaMax = 30000

        # First original objective function
        if self.obj == 0:
            return (1.10471 * x1 * x1 * x2) + (0.04811 * x3 * x4) * (14.0 + x2)
        elif self.obj == 1:
            # Second original objective function
            return (4 * P * L * L * L) / (E * x4 * x3 * x3 * x3)
        elif self.obj == 2:
            # Constraint functions
            M = P * (L + (x2 / 2))
            tmpVar = ((x2 * x2) / 4.0) + np.power((x1 + x3) / 2.0, 2)
            R = np.sqrt(tmpVar)
            tmpVar = ((x2 * x2) / 12.0) + np.power((x1 + x3) / 2.0, 2)
            J = 2 * np.sqrt(2) * x1 * x2 * tmpVar

            tauDashDash = (M * R) / J
            tauDash = P / (np.sqrt(2) * x1 * x2)
            tmpVar = tauDash * tauDash + ((2 * tauDash * tauDashDash * x2) / (2 * R)) + (tauDashDash * tauDashDash)
            tau = np.sqrt(tmpVar)
            sigma = (6 * P * L) / (x4 * x3 * x3)
            tmpVar = 4.013 * E * np.sqrt((x3 * x3 * x4 * x4 * x4 * x4 * x4 * x4) / 36.0) / (L * L)
            tmpVar2 = (x3 / (2 * L)) * np.sqrt(E / (4 * G))
            PC = tmpVar * (1 - tmpVar2)

            g0 = tauMax - tau
            g1 = sigmaMax - sigma
            g2 = x4 - x1
            g3 = PC - P
            g0 = np.where(g0 < 0, -g0, 0)
            g1 = np.where(g1 < 0, -g1, 0)
            g2 = np.where(g2 < 0, -g2, 0)
            g3 = np.where(g3 < 0, -g3, 0)
            return g0 + g1 + g2 + g3
        else:
            raise ValueError('obj')
            

class RE33(Objective):
    def __init__(self, num, ns=None, obj=1, transport='ipc', args=None, verbose=False):
        obj = obj - 1
        n_var = 4
        n_obj = 3
        bounds = list(zip([55.0, 75.0, 1000.0, 11.0], [80.0, 110.0, 3000.0, 20.0]))
        super(RE33, self).__init__(num, n_var, n_obj, bounds, obj, ns, transport, args, verbose)
        
    def call(self, x, *args):
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x4 = x[:, 3]

        # First original objective function
        if self.obj == 0:
            return 4.9 * 1e-5 * (x2 * x2 - x1 * x1) * (x4 - 1.0)
        elif self.obj == 1:
            # Second original objective function
            return ((9.82 * 1e6) * (x2 * x2 - x1 * x1)) / (x3 * x4 * (x2 * x2 * x2 - x1 * x1 * x1))
        elif self.obj == 2:
            # Reformulated objective functions
            g0 = (x2 - x1) - 20.0
            g1 = 0.4 - (x3 / (3.14 * (x2 * x2 - x1 * x1)))
            g2 = 1.0 - (2.22 * 1e-3 * x3 * (x2 * x2 * x2 - x1 * x1 * x1)) / np.power((x2 * x2 - x1 * x1), 2)
            g3 = (2.66 * 1e-2 * x3 * x4 * (x2 * x2 * x2 - x1 * x1 * x1)) / (x2 * x2 - x1 * x1) - 900.0
            g0 = np.where(g0 < 0, -g0, 0)
            g1 = np.where(g1 < 0, -g1, 0)
            g2 = np.where(g2 < 0, -g2, 0)
            g3 = np.where(g3 < 0, -g3, 0)

            return g0 + g1 + g2 + g3
        else:
            raise ValueError('obj')
            

class RE34(Objective):
    def __init__(self, num, ns=None, obj=1, transport='ipc', args=None, verbose=False):
        obj = obj - 1
        n_var = 5
        n_obj = 3
        bounds = list(zip([1.0, 1.0, 1.0, 1.0, 1.0], [3.0, 3.0, 3.0, 3.0, 3.0]))
        super(RE34, self).__init__(num, n_var, n_obj, bounds, obj, ns, transport, args, verbose)
        
    def call(self, x, *args):
        if self.obj == 0:
            return 1640.2823 + (2.3573285 * x[:, 0]) + (2.3220035 * x[:, 1]) + (4.5688768 * x[:, 2]) + (
                        7.7213633 * x[:, 3]) + (4.4559504 * x[:, 4])
        elif self.obj == 1:
            return 6.5856 + (1.15 * x[:, 0]) - (1.0427 * x[:, 1]) + (0.9738 * x[:, 2]) + (0.8364 * x[:, 3]) - (
                        0.3695 * x[:, 0] * x[:, 3]) + (0.0861 * x[:, 0] * x[:, 4]) + (0.3628 * x[:, 1] * x[:, 3]) - (
                               0.1106 * x[:, 0] * x[:, 0]) - (0.3437 * x[:, 2] * x[:, 2]) + (0.1764 * x[:, 3] * x[:, 3])
        elif self.obj == 2:
            return -0.0551 + (0.0181 * x[:, 0]) + (0.1024 * x[:, 1]) + (0.0421 * x[:, 2]) - (
                        0.0073 * x[:, 0] * x[:, 1]) + (0.024 * x[:, 1] * x[:, 2]) - (0.0118 * x[:, 1] * x[:, 3]) - (
                               0.0204 * x[:, 2] * x[:, 3]) - (0.008 * x[:, 2] * x[:, 4]) - (
                               0.0241 * x[:, 1] * x[:, 1]) + (0.0109 * x[:, 3] * x[:, 3])
        else:
            raise ValueError('obj')
            
            
class RE35(Objective):
    def __init__(self, num, ns=None, obj=1, transport='ipc', args=None, verbose=False):
        obj = obj - 1
        n_var = 7
        n_obj = 3
        bounds = list(zip([2.6, 0.7, 17.0, 7.3, 7.3, 2.9, 5.0], [3.6, 0.8, 28.0, 8.3, 8.3, 3.9, 5.5]))
        super(RE35, self).__init__(num, n_var, n_obj, bounds, obj, ns, transport, args, verbose)
        
    def call(self, x, *args):
        x1 = x[:,0]
        x2 = x[:,1]
        x3 = np.round(x[:,2])
        x4 = x[:,3]
        x5 = x[:,4]
        x6 = x[:,5]
        x7 = x[:,6]
        
        if self.obj == 0:
            # First original objective function (weight)
            #time.sleep(0.02)
            return 0.7854 * x1 * (x2 * x2) * (((10.0 * x3 * x3) / 3.0) + (14.933 * x3) - 43.0934) - 1.508 * x1 * (x6 * x6 + x7 * x7) + 7.477 * (x6 * x6 * x6 + x7 * x7 * x7) + 0.7854 * (x4 * x6 * x6 + x5 * x7 * x7)
        elif self.obj == 1:
            #time.sleep(0.01)
            # Second original objective function (stress)
            tmpVar = np.power((745.0 * x4) / (x2 * x3), 2.0)  + 1.69 * 1e7
            return np.sqrt(tmpVar) / (0.1 * x6 * x6 * x6)
        elif self.obj == 2:
            #time.sleep(0.1)
            # Constraint functions 	
            tmpVar = np.power((745.0 * x4) / (x2 * x3), 2.0)  + 1.69 * 1e7
            f1 = np.sqrt(tmpVar) / (0.1 * x6 * x6 * x6)
            g0 = -(1.0 / (x1 * x2 * x2 * x3)) + 1.0 / 27.0
            g1 = -(1.0 / (x1 * x2 * x2 * x3 * x3)) + 1.0 / 397.5
            g2 = -(x4 * x4 * x4) / (x2 * x3 * x6 * x6 * x6 * x6) + 1.0 / 1.93
            g3 = -(x5 * x5 * x5) / (x2 * x3 * x7 * x7 * x7 * x7) + 1.0 / 1.93
            g4 = -(x2 * x3) + 40.0
            g5 = -(x1 / x2) + 12.0
            g6 = -5.0 + (x1 / x2)
            g7 = -1.9 + x4 - 1.5 * x6
            g8 = -1.9 + x5 - 1.1 * x7
            g9 =  -f1 + 1300.0
            tmpVar = np.power((745.0 * x5) / (x2 * x3), 2.0) + 1.575 * 1e8
            g10 = -np.sqrt(tmpVar) / (0.1 * x7 * x7 * x7) + 1100.0
            g0 = np.where(g0 < 0, -g0, 0)
            g1 = np.where(g1 < 0, -g1, 0) 
            g2 = np.where(g2 < 0, -g2, 0) 
            g3 = np.where(g3 < 0, -g3, 0) 
            g4 = np.where(g4 < 0, -g4, 0) 
            g5 = np.where(g5 < 0, -g5, 0) 
            g6 = np.where(g6 < 0, -g6, 0) 
            g7 = np.where(g7 < 0, -g7, 0) 
            g8 = np.where(g8 < 0, -g8, 0) 
            g9 = np.where(g9 < 0, -g9, 0) 
            g10 = np.where(g10 < 0, -g10, 0) 
            return g0 + g1 + g2 + g3 + g4 + g5 + g6 + g7 + g8 + g9 + g10
        else:
            raise ValueError('obj')
            

class RE36(Objective):
    def __init__(self, num, obj=1, ns=None, transport='ipc', args=None, verbose=False):
        obj = obj - 1
        n_var = 4
        n_obj = 3
        bounds = list(zip([12.0, 12.0, 12.0, 12.0], [60.0, 60.0, 60.0, 60.0]))
        super(RE36, self).__init__(num, n_var, n_obj, bounds, obj, ns, transport, args, verbose)
        
    def call(self, x, *args):
        # all the four variables must be inverger values
        x1 = np.round(x[:,0])
        x2 = np.round(x[:,1])
        x3 = np.round(x[:,2])
        x4 = np.round(x[:,3])

        if self.obj == 0:
            # First original objective function
            #time.sleep(0.01)
            return np.abs(6.931 - ((x3 / x1) * (x4 / x2)))
        elif self.obj == 1:
            # Second original objective function (the maximum value among the four variables)
            #time.sleep(0.012)
            return np.max(x, axis=1)
        elif self.obj == 2:
            #time.sleep(0.017)
            f0 = np.abs(6.931 - ((x3 / x1) * (x4 / x2)))
            g0 = 0.5 - (f0 / 6.931)    
            g0 = np.where(g0 < 0, -g0, 0)                
            return g0
        else:
            raise ValueError('obj')
            
            
class RE37(Objective):
    def __init__(self, num, ns=None, obj=1, transport='ipc', args=None, verbose=False):
        obj = obj - 1
        n_var = 4
        n_obj = 3
        bounds = list(zip([0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]))
        super(RE37, self).__init__(num, n_var, n_obj, bounds, obj, ns, transport, args, verbose)
        
    def call(self, x, *args):
        xAlpha = x[:,0]
        xHA = x[:,1]
        xOA = x[:,2]
        xOPTT = x[:,3]

        if self.obj == 0:
            # f1 (TF_max)
            return 0.692 + (0.477 * xAlpha) - (0.687 * xHA) - (0.080 * xOA) - (0.0650 * xOPTT) - (0.167 * xAlpha * xAlpha) - (0.0129 * xHA * xAlpha) + (0.0796 * xHA * xHA) - (0.0634 * xOA * xAlpha) - (0.0257 * xOA * xHA) + (0.0877 * xOA * xOA) - (0.0521 * xOPTT * xAlpha) + (0.00156 * xOPTT * xHA) + (0.00198 * xOPTT * xOA) + (0.0184 * xOPTT * xOPTT)
        elif self.obj == 1:
            # f2 (X_cc)
            return 0.153 - (0.322 * xAlpha) + (0.396 * xHA) + (0.424 * xOA) + (0.0226 * xOPTT) + (0.175 * xAlpha * xAlpha) + (0.0185 * xHA * xAlpha) - (0.0701 * xHA * xHA) - (0.251 * xOA * xAlpha) + (0.179 * xOA * xHA) + (0.0150 * xOA * xOA) + (0.0134 * xOPTT * xAlpha) + (0.0296 * xOPTT * xHA) + (0.0752 * xOPTT * xOA) + (0.0192 * xOPTT * xOPTT)
        elif self.obj == 2:
            # f3 (TT_max
            return 0.370 - (0.205 * xAlpha) + (0.0307 * xHA) + (0.108 * xOA) + (1.019 * xOPTT) - (0.135 * xAlpha * xAlpha) + (0.0141 * xHA * xAlpha) + (0.0998 * xHA * xHA) + (0.208 * xOA * xAlpha) - (0.0301 * xOA * xHA) - (0.226 * xOA * xOA) + (0.353 * xOPTT * xAlpha) - (0.0497 * xOPTT * xOA) - (0.423 * xOPTT * xOPTT) + (0.202 * xHA * xAlpha * xAlpha) - (0.281 * xOA * xAlpha * xAlpha) - (0.342 * xHA * xHA * xAlpha) - (0.245 * xHA * xHA * xOA) + (0.281 * xOA * xOA * xHA) - (0.184 * xOPTT * xOPTT * xAlpha) - (0.281 * xHA * xAlpha * xOA)
        else:
            raise ValueError('obj')
            
            
class RE41(Objective):
    def __init__(self, num, ns=None, obj=1, transport='ipc', args=None, verbose=False):
        obj = obj - 1
        n_var = 7
        n_obj = 4
        bounds = list(zip([0.5, 0.45, 0.5, 0.5, 0.875, 0.4, 0.4], [1.5, 1.35, 1.5, 1.5, 2.625, 1.2, 1.2]))
        super(RE41, self).__init__(num, n_var, n_obj, bounds, obj, ns, transport, args, verbose)
        
    def call(self, x, *args):
        x1 = x[:,0]
        x2 = x[:,1]
        x3 = x[:,2]
        x4 = x[:,3]
        x5 = x[:,4]
        x6 = x[:,5]
        x7 = x[:,6]

        if self.obj == 0:
            # First original objective function
            return 1.98 + 4.9 * x1 + 6.67 * x2 + 6.98 * x3 + 4.01 * x4 + 1.78 * x5 + 0.00001 * x6 + 2.73 * x7

        elif self.obj == 1:
            # Second original objective function
            return 4.72 - 0.5 * x4 - 0.19 * x2 * x3

        elif self.obj == 2:
            # Third original objective function
            Vmbp = 10.58 - 0.674 * x1 * x2 - 0.67275 * x2
            Vfd = 16.45 - 0.489 * x3 * x7 - 0.843 * x5 * x6
            return 0.5 * (Vmbp + Vfd)
        elif self.obj == 3:
            # Constraint functions
            g0 = 1 -(1.16 - 0.3717 * x2 * x4 - 0.0092928 * x3)
            g1 = 0.32 -(0.261 - 0.0159 * x1 * x2 - 0.06486 * x1 -  0.019 * x2 * x7 + 0.0144 * x3 * x5 + 0.0154464 * x6)
            g2 = 0.32 -(0.214 + 0.00817 * x5 - 0.045195 * x1 - 0.0135168 * x1 + 0.03099 * x2 * x6 - 0.018 * x2 * x7 + 0.007176 * x3 + 0.023232 * x3 - 0.00364 * x5 * x6 - 0.018 * x2 * x2)
            g3 = 0.32 -(0.74 - 0.61 * x2 - 0.031296 * x3 - 0.031872 * x7 + 0.227 * x2 * x2)
            g4 = 32 -(28.98 + 3.818 * x3 - 4.2 * x1 * x2 + 1.27296 * x6 - 2.68065 * x7)
            g5 = 32 -(33.86 + 2.95 * x3 - 5.057 * x1 * x2 - 3.795 * x2 - 3.4431 * x7 + 1.45728)
            g6 =  32 -(46.36 - 9.9 * x2 - 4.4505 * x1)
            Vmbp = 10.58 - 0.674 * x1 * x2 - 0.67275 * x2
            Vfd = 16.45 - 0.489 * x3 * x7 - 0.843 * x5 * x6
            f1 = 4.72 - 0.5 * x4 - 0.19 * x2 * x3
            g7 =  4 - f1
            g8 =  9.9 - Vmbp
            g9 =  15.7 - Vfd

            g0 = np.where(g0 < 0, -g0, 0)
            g1 = np.where(g1 < 0, -g1, 0) 
            g2 = np.where(g2 < 0, -g2, 0) 
            g3 = np.where(g3 < 0, -g3, 0) 
            g4 = np.where(g4 < 0, -g4, 0) 
            g5 = np.where(g5 < 0, -g5, 0) 
            g6 = np.where(g6 < 0, -g6, 0) 
            g7 = np.where(g7 < 0, -g7, 0) 
            g8 = np.where(g8 < 0, -g8, 0) 
            g9 = np.where(g9 < 0, -g9, 0)
            return g0 + g1 + g2 + g3 + g4 + g5 + g6 + g7 + g8 + g9
        else:
            raise ValueError('obj')
            
            
class RE42(Objective):
    def __init__(self, num, ns=None, obj=1, transport='ipc', args=None, verbose=False):
        obj = obj - 1
        n_var = 6
        n_obj = 4
        bounds = list(zip([150.0, 20.0, 13.0, 10.0, 14.0, 0.63], [274.32, 32.31, 25.0, 11.71, 18.0, 0.75]))
        super(RE42, self).__init__(num, n_var, n_obj, bounds, obj, ns, transport, args, verbose)
        
    def call(self, x, *args):
        x_L = x[:,0]
        x_B = x[:,1]
        x_D = x[:,2]
        x_T = x[:,3]
        x_Vk = x[:,4]
        x_CB = x[:,5]
   
        displacement = 1.025 * x_L * x_B * x_T * x_CB
        V = 0.5144 * x_Vk
        g = 9.8065
        Fn = V / np.power(g * x_L, 0.5)
        a = (4977.06 * x_CB * x_CB) - (8105.61 * x_CB) + 4456.51
        b = (-10847.2 * x_CB * x_CB) + (12817.0 * x_CB) - 6960.32

        power = (np.power(displacement, 2.0/3.0) * np.power(x_Vk, 3.0)) / (a + (b * Fn))
        outfit_weight = 1.0 * np.power(x_L , 0.8) * np.power(x_B , 0.6) * np.power(x_D, 0.3) * np.power(x_CB, 0.1)
        steel_weight = 0.034 * np.power(x_L ,1.7) * np.power(x_B ,0.7) * np.power(x_D ,0.4) * np.power(x_CB ,0.5)
        machinery_weight = 0.17 * np.power(power, 0.9)
        light_ship_weight = steel_weight + outfit_weight + machinery_weight

        ship_cost = 1.3 * ((2000.0 * np.power(steel_weight, 0.85))  + (3500.0 * outfit_weight) + (2400.0 * np.power(power, 0.8)))
        capital_costs = 0.2 * ship_cost

        DWT = displacement - light_ship_weight

        running_costs = 40000.0 * np.power(DWT, 0.3)

        round_trip_miles = 5000.0
        sea_days = (round_trip_miles / 24.0) * x_Vk
        handling_rate = 8000.0

        daily_consumption = ((0.19 * power * 24.0) / 1000.0) + 0.2
        fuel_price = 100.0
        fuel_cost = 1.05 * daily_consumption * sea_days * fuel_price
        port_cost = 6.3 * np.power(DWT, 0.8)

        fuel_carried = daily_consumption * (sea_days + 5.0)
        miscellaneous_DWT = 2.0 * np.power(DWT, 0.5)
        
        cargo_DWT = DWT - fuel_carried - miscellaneous_DWT
        port_days = 2.0 * ((cargo_DWT / handling_rate) + 0.5)
        RTPA = 350.0 / (sea_days + port_days)

        voyage_costs = (fuel_cost + port_cost) * RTPA
        annual_costs = capital_costs + running_costs + voyage_costs
        annual_cargo = cargo_DWT * RTPA

        if self.obj == 0:
            return annual_costs / annual_cargo
        elif self.obj == 1:
            return light_ship_weight
        elif self.obj == 2:
            # f_2 is dealt as a minimization problem
            return -annual_cargo
        elif self.obj == 3:
            # Reformulated objective functions
            constraintFuncs0 = (x_L / x_B) - 6.0
            constraintFuncs1 = -(x_L / x_D) + 15.0
            constraintFuncs2 = -(x_L / x_T) + 19.0
            constraintFuncs3 = 0.45 * np.power(DWT, 0.31) - x_T
            constraintFuncs4 = 0.7 * x_D + 0.7 - x_T
            constraintFuncs5 = 500000.0 - DWT
            constraintFuncs6 = DWT - 3000.0
            constraintFuncs7 = 0.32 - Fn

            KB = 0.53 * x_T
            BMT = ((0.085 * x_CB - 0.002) * x_B * x_B) / (x_T * x_CB)
            KG = 1.0 + 0.52 * x_D
            constraintFuncs8 = (KB + BMT - KG) - (0.07 * x_B)
            
            constraintFuncs0 = np.where(constraintFuncs0 < 0, -constraintFuncs0, 0)
            constraintFuncs1 = np.where(constraintFuncs1 < 0, -constraintFuncs1, 0)
            constraintFuncs2 = np.where(constraintFuncs2 < 0, -constraintFuncs2, 0)
            constraintFuncs3 = np.where(constraintFuncs3 < 0, -constraintFuncs3, 0)
            constraintFuncs4 = np.where(constraintFuncs4 < 0, -constraintFuncs4, 0)
            constraintFuncs5 = np.where(constraintFuncs5 < 0, -constraintFuncs5, 0)
            constraintFuncs6 = np.where(constraintFuncs6 < 0, -constraintFuncs6, 0)
            constraintFuncs7 = np.where(constraintFuncs7 < 0, -constraintFuncs7, 0)
            constraintFuncs8 = np.where(constraintFuncs8 < 0, -constraintFuncs8, 0)
             
            return constraintFuncs0 + constraintFuncs1 + constraintFuncs2 + constraintFuncs3 + constraintFuncs4 + constraintFuncs5 + constraintFuncs6 + constraintFuncs7 + constraintFuncs8
        else:
            raise ValueError('obj')
            
            
class RE61(Objective):
    def __init__(self, num, ns=None, obj=1, transport='ipc', args=None, verbose=False):
        obj = obj - 1
        n_var = 3
        n_obj = 6
        bounds = list(zip([0.01, 0.01, 0.01], [0.45, 0.1, 0.1]))
        super(RE61, self).__init__(num, n_var, n_obj, bounds, obj, ns, transport, args, verbose)
        
    def call(self, x, *args):
        if self.obj == 0:
            return 106780.37 * (x[:, 1] + x[:, 2]) + 61704.67
        elif self.obj == 1:
            return 3000 * x[:, 0]
        elif self.obj == 2:
            return 305700 * 2289 * x[:, 1] / np.power(0.06*2289, 0.65)
        elif self.obj == 3:
            return 250 * 2289 * np.exp(-39.75*x[:, 1]+9.9*x[:, 2]+2.74)
        elif self.obj == 4:
            return 25 * (1.39 /(x[:, 0]*x[:, 1]) + 4940*x[:, 2] -80)
        elif self.obj == 5:
            g0 = 1 - (0.00139 / (x[:, 0] * x[:, 1]) + 4.94 * x[:, 2] - 0.08)
            g1 = 1 - (0.000306 / (x[:, 0] * x[:, 1]) + 1.082 * x[:, 2] - 0.0986)
            g2 = 50000 - (12.307 / (x[:, 0] * x[:, 1]) + 49408.24 * x[:, 2] + 4051.02)
            g3 = 16000 - (2.098 / (x[:, 0] * x[:, 1]) + 8046.33 * x[:, 2] - 696.71)
            g4 = 10000 - (2.138 / (x[:, 0] * x[:, 1]) + 7883.39 * x[:, 2] - 705.04)
            g5 = 2000 - (0.417 * x[:, 0] * x[:, 1] + 1721.26 * x[:, 2] - 136.54)
            g6 = 550 - (0.164 / (x[:, 0] * x[:, 1]) + 631.13 * x[:, 2] - 54.48)
            g0 = np.where(g0 < 0, -g0, 0)
            g1 = np.where(g1 < 0, -g1, 0)
            g2 = np.where(g2 < 0, -g2, 0)
            g3 = np.where(g3 < 0, -g3, 0)
            g4 = np.where(g4 < 0, -g4, 0)
            g5 = np.where(g5 < 0, -g5, 0)
            g6 = np.where(g6 < 0, -g6, 0)
            return g0 + g1 + g2 + g3 + g4 + g5 + g6
        else:
            raise ValueError('obj')
            
            
class RE91(Objective):
    def __init__(self, num, ns=None, obj=1, transport='ipc', args=None, verbose=False):
        obj = obj - 1
        n_var = 7
        n_obj = 9
        bounds = list(zip([0.5, 0.45, 0.5, 0.5, 0.875, 0.4, 0.4], [1.5, 1.35, 1.5, 1.5, 2.625, 1.2, 1.2]))
        super(RE91, self).__init__(num, n_var, n_obj, bounds, obj, ns, transport, args, verbose)
        
    def call(self, x, *args):
        x1 = x[:,0]
        x2 = x[:,1]
        x3 = x[:,2]
        x4 = x[:,3]
        x5 = x[:,4]
        x6 = x[:,5]
        x7 = x[:,6]
        # stochastic variables
        x8 = 0.006 * (np.random.normal(0, 1)) + 0.345
        x9 = 0.006 * (np.random.normal(0, 1)) + 0.192
        x10 = 10 * (np.random.normal(0, 1)) + 0.0
        x11 = 10 * (np.random.normal(0, 1)) + 0.0

        if self.obj == 0:
            # First function
            return 1.98 + 4.9 * x1 + 6.67 * x2 + 6.98 * x3 +  4.01 * x4 +  1.75 * x5 +  0.00001 * x6  +  2.73 * x7
        elif self.obj == 1:
            # Second function
            f = (1.16 - 0.3717* x2 * x4 - 0.00931 * x2 * x10 - 0.484 * x3 * x9 + 0.01343 * x6 * x10 )/1.0
            return np.where(f > 0, f, 0) 
        elif self.obj == 2:
            # Third function
            f = (0.261 - 0.0159 * x1 * x2 - 0.188 * x1 * x8 - 0.019 * x2 * x7 + 0.0144 * x3 * x5 + 0.87570001 * x5 * x10 + 0.08045 * x6 * x9 + 0.00139 * x8 * x11 + 0.00001575 * x10 * x11)/0.32
            return np.where(f>0, f, 0)
        elif self.obj == 3:
            # Fourth function
            f = (0.214 + 0.00817 * x5 - 0.131 * x1 * x8 - 0.0704 * x1 * x9 + 0.03099 * x2 * x6 - 0.018 * x2 * x7 + 0.0208 * x3 * x8 + 0.121 * x3 * x9 - 0.00364 * x5 * x6 + 0.0007715 * x5 * x10 - 0.0005354 * x6 * x10 + 0.00121 * x8 * x11 + 0.00184 * x9 * x10 - 0.018 * x2 * x2)/0.32
            return np.where(f>0, f, 0)
        elif self.obj == 4:
            # Fifth function  
            f = (0.74 - 0.61* x2 - 0.163 * x3 * x8 + 0.001232 * x3 * x10 - 0.166 * x7 * x9 + 0.227 * x2 * x2)/0.32
            return np.where(f>0, f, 0)
        elif self.obj == 5:
            # Sixth function       
            tmp = (( 28.98 + 3.818 * x3 - 4.2 * x1 * x2 + 0.0207 * x5 * x10 + 6.63 * x6 * x9 - 7.77 * x7 * x8 + 0.32 * x9 * x10) + (33.86 + 2.95 * x3 + 0.1792 * x10 - 5.057 * x1 * x2 - 11 * x2 * x8 - 0.0215 * x5 * x10 - 9.98 * x7 * x8 + 22 * x8 * x9) + (46.36 - 9.9 * x2 - 12.9 * x1 * x8 + 0.1107 * x3 * x10) )/3
            f = tmp/32
            return np.where(f>0, f, 0) 
        elif self.obj == 6:
            # Seventh function
            f = (4.72 - 0.5 * x4 - 0.19 * x2 * x3 - 0.0122 * x4 * x10 + 0.009325 * x6 * x10 + 0.000191 * x11 * x11)/4.0
            return np.where(f>0, f, 0)
        elif self.obj == 7:
            # EighthEighth function   
            f = (10.58 - 0.674 * x1 * x2 - 1.95  * x2 * x8  + 0.02054  * x3 * x10 - 0.0198  * x4 * x10  + 0.028  * x6 * x10)/9.9               
            return np.where(f>0, f, 0)
        elif self.obj == 8:
            f = (16.45 - 0.489 * x3 * x7 - 0.843 * x5 * x6 + 0.0432 * x9 * x10 - 0.0556 * x9 * x11 - 0.000786 * x11 * x11)/15.7
            # Ninth function               
            return np.where(f>0, f, 0)
        else:
            raise ValueError('obj')