import numpy as np
import copy
from statistics import correlation
from UQpy.sampling import LatinHypercubeSampling
from Ishigami_functions import estimate_sobol, sobol_mc, generate_tensor_c

import matplotlib.pyplot as plt
import matplotlib

font = {'family': 'serif',
        'size': 13}

matplotlib.rc('font', **font)
matplotlib.rc('xtick', labelsize=15)
matplotlib.rc('ytick', labelsize=15)


# Define the model inputs
def f_fun(z, a=5, b=0.1):
    z1 = z[0]
    z2 = z[1]
    z3 = z[2]
    return np.sin(z1) + a * (np.sin(z2) ** 2) + b * np.sin(z1) * (z3 ** 4)


def V1_analytic(b):
    return 0.5 * (1 + 1 / 5 * b * np.pi ** 4) ** 2


def V2_analytic(a):
    return (a ** 2) / 8


def V13_analytic(b):
    return np.pi ** 8 * b ** 2 * (1 / 18 - 1 / 50)


def Vt_analytic(a, b):
    return 0.5 + 1 / 8 * a ** 2 + 1 / 5 * np.pi ** 4 * b + 1 / 18 * b ** 2 * np.pi ** 8


N = 10 ** 5
d = 3

# sample 2N of the input parameter.
A = np.random.uniform(low=-np.pi, high=np.pi, size=(N, d))
B = np.random.uniform(low=-np.pi, high=np.pi, size=(N, d))

# generate matrix C=[C1, C2, C3].
C = generate_tensor_c(A=A, B=B, d=3)

# evaluate the model for given samples.
YA, YB, YC = sobol_mc(A=A, B=B, C=C, f=f_fun)

# estimate sobol indices.
main_effect, total_effect = estimate_sobol(YA, YB, YC, type="sobol")
