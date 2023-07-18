import numpy as np

# Uncomment these two lines if numba package is installed and make things happen much faster:
from numba import jit
@jit(nopython=True)
def RR_func(zc, k, eps):

    a = 1 / (1 - np.max(k)) + eps
    b = 1 / (1 - np.min(k)) - eps

    max_iter = 200  # use enough iterations for V to converge
    for i in range(1, max_iter):
        V = 0.5 * (a + b)
        r = np.sum(zc * (k - 1) / (V * (k - 1) + 1))
        if abs(r) < 1e-12:
            break

        if r > 0:
            a = V
        else:
            b = V

    if i >= max_iter:
        print("Flash warning!!!")

    x = zc / (V * (k - 1) + 1)
    y = k * x

    return (x, y, V)

class Flash:
    def __init__(self, components, ki, min_z=1e-11):
        self.components = components
        self.min_z = min_z
        self.ki = np.array(ki)

    def evaluate(self, pressure, zc):

        (x, y, V) = self.RR(zc, self.ki)
        return [y, x], [V, 1-V]

    def RR(self, zc, k):
        return RR_func(zc, k, self.min_z)


class Density:
    def __init__(self, dens0=1000, compr=0, p0=1, x_mult=0):
        self.compr = compr
        self.p0 = p0
        self.dens0 = dens0
        self.x_max = x_mult

    def evaluate(self, pressure, x_co2):
        density = (self.dens0 + x_co2 * self.x_max) * (1 + self.compr * (pressure - self.p0))
        return density

class ViscosityConst:
    def __init__(self, visc):
        self.visc = visc

    def evaluate(self):
        return self.visc

class Enthalpy:
    def __init__(self, tref=273.15, hcap=0.0357):
        self.tref = tref
        self.hcap = hcap

    def evaluate(self, temp):
        # methane heat capacity
        enthalpy = self.hcap * (temp - self.tref)
        return enthalpy

class PhaseRelPerm:
    def __init__(self, phase, swc=0, sgr=0):
        self.phase = phase

        self.Swc = swc
        self.Sgr = sgr
        if phase == "oil":
            self.kre = 1
            self.sr = self.Swc
            self.sr1 = self.Sgr
            self.n = 2
        elif phase == 'gas':
            self.kre = 1
            self.sr = self.Sgr
            self.sr1 = self.Swc
            self.n = 2
        else:  # water
            self.kre = 1
            self.sr = 0
            self.sr1 = 0
            self.n = 1


    def evaluate(self, sat):

        if sat >= 1 - self.sr1:
            kr = self.kre

        elif sat <= self.sr:
            kr = 0

        else:
            # general Brook-Corey
            kr = self.kre * ((sat - self.sr) / (1 - self.Sgr - self.Swc)) ** self.n

        return kr


