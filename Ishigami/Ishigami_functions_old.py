import numpy as np
import copy


def generate_rvs(N, d):
    # step 1
    A = np.random.uniform(low=-np.pi, high=np.pi, size=(N, d))
    B = np.random.uniform(low=-np.pi, high=np.pi, size=(N, d))

    # step 2
    C1, C2, C3 = copy.deepcopy(B), copy.deepcopy(B), copy.deepcopy(B)
    C1[:, 0] = A[:, 0]
    C2[:, 1] = A[:, 1]
    C3[:, 2] = A[:, 2]

    return A, B, C1, C2, C3


def sobol_mc(A, B, C1, C2, C3, f):
    # step 3
    YA = f(z1=A[:, 0], z2=A[:, 1], z3=A[:, 2])
    YB = f(z1=B[:, 0], z2=B[:, 1], z3=B[:, 2])
    YC1 = f(z1=C1[:, 0], z2=C1[:, 1], z3=C1[:, 2])
    YC2 = f(z1=C2[:, 0], z2=C2[:, 1], z3=C2[:, 2])
    YC3 = f(z1=C3[:, 0], z2=C3[:, 1], z3=C3[:, 2])
    return YA, YB, YC1, YC2, YC3


def estimate_sobol(YA, YB, YC1, YC2, YC3):
    muA = np.mean(YA)
    varA = np.var(YA)
    N = len(YA)

    S_res, S_tot_res = np.zeros(3), np.zeros(3)

    # main or first Sobol' indices (Saltelli estimator)
    S_res[0] = (1 / (N - 1) * YA.T @ YC1 - muA ** 2) / varA
    S_res[1] = (1 / (N - 1) * YA.T @ YC2 - muA ** 2) / varA
    S_res[2] = (1 / (N - 1) * YA.T @ YC3 - muA ** 2) / varA

    # total effect Sobol' indices (Saltelli estimator)
    S_tot_res[0] = 1 - (1 / (N - 1) * YB.T @ YC1 - muA ** 2) / varA
    S_tot_res[1] = 1 - (1 / (N - 1) * YB.T @ YC2 - muA ** 2) / varA
    S_tot_res[2] = 1 - (1 / (N - 1) * YB.T @ YC3 - muA ** 2) / varA

    return S_res, S_tot_res