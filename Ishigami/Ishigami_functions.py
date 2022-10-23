import numpy as np
import copy


def generate_tensor_c(A, B, d):
    """Generate Random variables in a matrix A, B, C.

    :param d: (int) number of random variables.
    :param A: (ndarray) matrix of size (N x d).
    :param B: (ndarray) matrix of size (N x d).
    :return: (ndarray) C tensor, size (d x N x d).
    """
    # copy matrix B and only modify one column.
    C = np.zeros((d, np.shape(B)[0], np.shape(B)[1]))
    for ii in range(d):
        C[ii, :, :] = copy.deepcopy(B)
        C[ii, :, ii] = A[:, ii]
    return C


def sobol_mc(A, B, C, f):
    """Evaluate the QoI given input parameters.

    :param A: (ndarray) sampled matrix A.
    :param B: (ndarray) sampled matrix B.
    :param C: (ndarray) sampled matrix C.
    :param f: model function (mapping from input to output)-- output is scalar.
    :return: (ndarray) f(A) size (N x 1), (ndarray) f(B) size (N x 1), (ndarray) f(C) size (N x d)
    """
    # initialize matrices.
    N, d = np.shape(A)
    YA = np.zeros(N)
    YB = np.zeros(N)
    YC = np.zeros((N, d))

    # evaluate function for each sample.
    for ii in range(N):
        YA[ii] = f(z=A[ii, :])
        YB[ii] = f(z=B[ii, :])
        for jj in range(d):
            YC[ii, jj] = f(z=C[jj, ii, :])
    return YA, YB, YC


def estimate_sobol(YA, YB, YC, type="sobol"):
    """Estimate Sobol' indices (main effect) Vi/V and (total effect) Ti/T.

    :param YA: (ndarray) output of input sampled matrix A.
    :param YB: (ndarray) output of input sampled matrix B.
    :param YC: (ndarray) output of input sampled matrix C.
    :param type: (str) type of estimator ("sobol", "owen", "saltelli_I", "saltelli_II", "janon", etc...)
    :return: (ndarray) S (main effect) size (d x 1), (ndarray) T (total effect) size (d x 1)
    """
    main_effect = estimator_main_effect(YA=YA, YB=YB, YC=YC, N=len(YA), type=type)
    total_effect = estimator_total_effect(YA=YA, YB=YB, YC=YC, N=len(YA), type=type)
    return main_effect, total_effect


def estimator_main_effect(YA, YB, YC, N, type="sobol"):
    """Computes the main effect sobol indices.

    :param YA: (ndarray) output of sampled matrix A size (N x 1).
    :param YB: (ndarray) output of sampled matrix B size (N x 1).
    :param YC: (ndarray) output of sampled matrix C size (N x d).
    :param N: (int) number of samples.
    :param type: (str) type of estimator ("sobol", "owen", "saltelli_I", "saltelli_II", "janon", etc...)
    :return: (ndarray) sobol indices main effect, size (d x 1).
    """
    if type == "owen":
        """unbaised
        A. B. Owen, Variance components and generalized Sobol’ indices, SIAM/ASA Journal on Uncertainty
        Quantification, 1 (2013), pp. 19–41, https://doi.org/10.1137/120876782, http://dx.doi.org/10.1137/
        120876782, https://arxiv.org/abs/http://dx.doi.org/10.1137/120876782.
        """
        V = np.mean(YA ** 2) - np.mean(YA) ** 2
        return ((2 * N) / (2 * N - 1) * (1 / N * YA.T @ YC -
                                         ((np.mean(YA) + np.mean(YC)) / 2) ** 2 +
                                         (np.var(YA) + np.var(YC)) / (4 * N))) / V
    if type == "sobol":
        """unbiased
        A. Saltelli, Making best use of model evaluations to compute sensitivity indices, 
        Computer Physics Communications, 145 (2002), pp. 280 – 297.
        """
        f02 = np.mean(YA) ** 2
        return (1 / N * YA.T @ YC - f02) / np.var(YA)

    if type == "saltelli_I":
        """bias O(1/n)
        A. Saltelli, Making best use of model evaluations to compute sensitivity indices, 
        Computer Physics Communications, 145 (2002), pp. 280 – 297.
        """
        f02 = np.mean(YA) * np.mean(YB)
        V = np.mean(YA ** 2) - np.mean(YA) ** 2
        return (1 / (N - 1) * YA.T @ YC - f02) / V

    if type == "saltelli_II":
        """bias O(1/n)
        A. Saltelli, Making best use of model evaluations to compute sensitivity indices, 
        Computer Physics Communications, 145 (2002), pp. 280 – 297.
        """
        f02 = np.mean(YA * YB)
        V = np.mean(YA ** 2) - np.mean(YA) ** 2
        return (1 / (N - 1) * YA.T @ YC - f02) / V

    if type == "janon":
        """ bias O(1/n)
        A. Janon, T. Klein, A. Lagnoux, M. Nodet, and C. Prieur, Asymptotic normality and efficiency
        of two Sobol index estimators, ESAIM: Probability and Statistics, 18 (2014), pp. 342–364.
        """
        f02 = np.mean(np.append(YA, YB))
        return (1 / N * YA.T @ YC - f02) / (np.mean(YA ** 2) - f02)


def estimator_total_effect(YA, YB, YC, N, type="sobol"):
    """Computes the total effect sobol indices.

    :param YA: (ndarray) output of sampled matrix A size (N x 1).
    :param YB: (ndarray) output of sampled matrix B size (N x 1).
    :param YC: (ndarray) output of sampled matrix C size (N x d).
    :param N: number of samples.
    :param type: "sobol", "owen", "saltelli_I", "saltelli_II", "janon", etc...
    :return: (ndarray) sobol total effect indices, size (d times 1).
    """
    if type == "sobol":
        """unbiased
        A. Saltelli, Making best use of model evaluations to compute sensitivity indices, 
        Computer Physics Communications, 145 (2002), pp. 280 – 297.
        """
        f02 = np.mean(YA) ** 2
        return 1 - (1 / N * YB.T @ YC - f02) / np.var(YA)

    if type == "saltelli_I":
        """bias O(1/n)
        A. Saltelli, Making best use of model evaluations to compute sensitivity indices, 
        Computer Physics Communications, 145 (2002), pp. 280 – 297.
        """
        f02 = np.mean(YA) * np.mean(YB)
        return 1 - (1 / (N - 1) * YB.T @ YC - f02) / (np.mean(YA ** 2) - np.mean(YA) ** 2)

    if type == "saltelli_II":
        """bias O(1/n)
        A. Saltelli, Making best use of model evaluations to compute sensitivity indices, 
        Computer Physics Communications, 145 (2002), pp. 280 – 297.
        """
        f02 = np.mean(YA * YB)
        return 1 - (1 / (N - 1) * YB.T @ YC - f02) / (np.mean(YA ** 2) - np.mean(YA) ** 2)

    if type == "janon":
        """ bias O(1/n)
        A. Janon, T. Klein, A. Lagnoux, M. Nodet, and C. Prieur, Asymptotic normality and efficiency
        of two Sobol index estimators, ESAIM: Probability and Statistics, 18 (2014), pp. 342–364.
        """
        f02 = np.mean(np.append(YA, YB))
        return 1 - (1 / N * YB.T @ YC - f02) / (np.mean(YA ** 2) - f02)

    if type == "owen":
        """unbaised
        A. B. Owen, Variance components and generalized Sobol’ indices, SIAM/ASA Journal on Uncertainty
        Quantification, 1 (2013), pp. 19–41, https://doi.org/10.1137/120876782, http://dx.doi.org/10.1137/
        120876782, https://arxiv.org/abs/http://dx.doi.org/10.1137/120876782.
        """
        T = np.zeros(np.shape(YC)[1])
        for ii in range(np.shape(YC)[1]):
            T[ii] = (1 / (2 * N) * np.sum((YB - YC[:, ii]) ** 2)) / (np.mean(YA ** 2) - np.mean(YA) ** 2)
        return T
