import numpy as np
from pygamoo import Objective
import matlab.engine

nobjs = 3
nvars = 7
#bounds = list(zip([0.2, 0.2, 0.2, 0.2, 1.0, 1.2, 0.0], [0.5, 0.5, 0.5, 0.5, 2.2, 1.6, 1.0]))
bounds = list(zip([0.2, 0.2, 0.2, 0.2, 1.0, 1.0, 0.0], [0.6, 0.6, 0.6, 0.6, 2.2, 1.6, 1.0]))

class CableF1(Objective):
    def call(self, x, *args):
        l = x[:, 0]
        p = x[:, 1]
        b = x[:, 2]
        s = x[:, 3]
        Ac = np.round(x[:, 4], 1)
        mask_0 = Ac < 1.2
        mask_1 = np.logical_and(1.2 <= Ac, Ac < 1.4)
        mask_2 = np.logical_and(1.4 <= Ac, Ac < 1.6)
        mask_3 = np.logical_and(1.6 <= Ac, Ac < 1.8)
        mask_4 = np.logical_and(1.8 <= Ac, Ac < 2.0)
        mask_5 = Ac >= 2.0
        Ac[mask_0] = 1000
        Ac[mask_1] = 1200
        Ac[mask_2] = 1400
        Ac[mask_3] = 1600
        Ac[mask_4] = 1800
        Ac[mask_5] = 2000

        MAT = x[:, 6]
        MAT_a = MAT.copy()
        mask_0 = MAT < 0.3333
        mask_1 = np.logical_and(0.3333 <= MAT, MAT < 0.6666)
        mask_2 = MAT >= 0.6666
        MAT[mask_0] = 20000
        MAT[mask_1] = 125000
        MAT[mask_2] = 400000
        MAT_a[mask_0] = 5.7943
        MAT_a[mask_1] = 0.9271
        MAT_a[mask_2] = 0.2897
        AA = Ac * 1.e-6
        p1 = 1.042e+05
        p2 = 5437
        p3 = 0.1583
        p4 = 0.0343
        dc = p1 * AA ** 3 + p2 * AA ** 2 + p3 * AA + p4
        p1 = 1.39e+06
        p2 = -7083
        p3 = 14.45
        p4 = 0.000927
        AO = p1 * AA ** 3 + p2 * AA ** 2 + p3 * AA + p4
        d0 = np.sqrt((4 * AO / np.pi) + (dc * dc))
        Ab = (l + s) * (p + b) - 1.5 * np.pi * d0 * d0 * 0.25
        Aback = Ab
        a = MAT_a
        c = 841.85
        mult = a * np.exp(c * AA)
        costfactor = 1.5 * mult + Aback
        uc = MAT
        cost = costfactor * uc
        cost = cost/1e6
        return cost


class CableF2(Objective):
    def call(self, x, *args):
        I = x[:, 5] * 1000
        return -I #np.ones(x.shape[0])*1145 #-I


class CableF3(Objective):
    def __init__(self, num, qname, host, port, log_queue=None, args=None):
        self.matlab = True
        super(CableF3, self).__init__(num, qname, host, port, log_queue, args)

    def call(self, x, *args):
        # run FEM_Code(qv,Tcz,alfa,ll,ss,pp,bb,I,AA)
        qv = 15000
        Tcz = 30
        alfa = 10
        eng = args[0]
        x = x.astype(float)
        #Temps = np.zeros(x.shape[0])
        #eng = matlab.engine.start_matlab()
        X = []
        for i, sol in enumerate(x):
            ll = sol[0]
            ss = sol[3]
            pp = sol[1]
            bb = sol[2]
            # I = 1145
            I = sol[5] * 1000
            Acn = np.round(sol[4], 1)
            if Acn < 1.2:
                Ac = 1000
            elif 1.2 <= Acn < 1.4:
                Ac = 1200
            elif 1.4 <= Acn < 1.6:
                Ac = 1400
            elif 1.6 <= Acn < 1.8:
                Ac = 1600
            elif 1.8 <= Acn < 2.0:
                Ac = 1800
            else:
                Ac = 2000
            AA = Ac * 1.e-6
            mat = sol[6]

            lambda_s = 0.8
            if mat < 0.33:
                lambda_b = 1.0
            elif 0.33 <= mat < 0.66:
                lambda_b = 1.54
            else:
                lambda_b = 3.0
            X.append([float(ll), float(pp), float(bb), float(ss), float(I), float(AA), float(lambda_s), float(lambda_b)])
            # run matlab
            #Temps[i] = eng.Compute_Temp2(*X, nargout=1)
        X = matlab.double(X)
        Temps = eng.Compute_Temp2(X, nargout=1)
        Temps = np.array(Temps).flatten()
        #Temps = np.where(Temps > 90, Temps**2, Temps)
        return Temps

    #def close(self):
    #    #self.eng.quit()
    #    super(CableF3, self).close()


class CableF3v2(Objective):
    def __init__(self, num, qname, host, port, log_queue=None):
        self.eng = matlab.engine.start_matlab()
        super(CableF3v2, self).__init__(num, qname, host, port, log_queue)

    def call(self, x):
        # run FEM_Code(qv,Tcz,alfa,ll,ss,pp,bb,I,AA)
        qv = 15000
        Tcz = 30
        alfa = 10
        x = x.astype(float)
        Temps = np.zeros(x.shape[0])
        for i, sol in enumerate(x):
            ll = sol[0]
            ss = sol[3]
            pp = sol[1]
            bb = sol[2]
            I = sol[5] * 1000
            Acn = np.round(sol[4], 1)
            if Acn < 1.2:
                Ac = 1000
            elif 1.2 <= Acn < 1.4:
                Ac = 1200
            elif 1.4 <= Acn < 1.6:
                Ac = 1400
            elif 1.6 <= Acn < 1.8:
                Ac = 1600
            elif 1.8 <= Acn < 2.0:
                Ac = 1800
            else:
                Ac = 2000
            AA = Ac * 1.e-6
            mat = sol[6]

            lambda_s = 0.8
            if mat < 0.33:
                lambda_b = 1.0
            elif 0.33 <= mat < 0.66:
                lambda_b = 1.54
            else:
                lambda_b = 3.0
            X = (float(ll), float(pp), float(bb), float(ss), float(I), float(AA), float(lambda_s), float(lambda_b))
            # run matlab
            Temps[i] = self.eng.Compute_Temp2(*X, nargout=1)
        return Temps

    def close(self):
        self.eng.quit()
        super(CableF3, self).close()


