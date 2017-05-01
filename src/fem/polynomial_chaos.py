import csv
import numpy as np
import scipy.integrate as integrate

from itertools import product
from functools import reduce
from scipy.interpolate import BarycentricInterpolator
from scipy.special import legendre

from fem.oned_deterministic import hat_basis


def read_eigendata(filename):
    """
    Given the filepath to the CSV file containing the eigenvalue/eigenfunction
    data from Matlab read it and reconstruct the interpolated functions as
    used by chebfun
    """

    with open(filename) as f:
        reader = csv.reader(f)
        rows = [row for row in reader]

    # Eigenvalues are stored in the first row of data
    lambdas = [float(l) for l in rows[0]]

    data = list()

    # Each eigenfunction is represented by nodes x_i and the values
    # at those nodes f(x_i). Each subsequent pairs of rows are the xi
    # and f(xi) of the function.
    for i in range(len(lambdas)):

        # Reconstruct the representation as used by chebfun
        p = BarycentricInterpolator([float(x) for x in rows[2*i + 1]],
                                    [float(y) for y in rows[2*i + 2]])

        data.append({'lambda': lambdas[i],
                     'beta': p})

    # Finally reverse the list so that we have the largest eigenvalue first
    data.reverse()

    return data


def prod(it):
    """
    Given it, an iterable return the product of all its entries
    """
    return reduce(lambda x, y: x*y, it, 1)


def mk_lengendre_basis(index):
    """
    Given the multi index 'index' construct a function representing
    that particular legendre basis function
    """

    # How many dimensions?
    d = len(index)

    # Construct the function
    def Ps(*args):

        # Add some error checking
        if len(*args) != d:
            raise ValueError('Expected %i arguments!' % d)

        return prod([legendre(n)(x) for n, x in zip(index, *args)])

    # Also add an index field so we can find out which ploynomial it is
    Ps._index = index

    return Ps


def legendre_chaos(d, p):
    """
    Given the dimensionality d and the highest degree of polynomial p
    return the P legendre polynomial basis functions
    """

    # Given p and d, generate all the multi-indicies
    indices = filter(lambda idx: sum(idx) <= p,
                     product(*tuple(range(p+1) for _ in range(d))))

    # Construct the corresponding legendre basis and return
    return [mk_lengendre_basis(idx) for idx in indices]


def eval_chi_s_squared(basis, s):
    """
    Given the legendre basis and an index s, compute the quantity
    <<\chi_s>>^2
    """

    Ps = basis[s]
    d = len(Ps._index)

    # Contrstuct the d [-1, 1] integration endpoints
    ranges = [[-1, 1] for _ in range(d)]

    # Do the integration
    res, _ = integrate.nquad(lambda *args: 0.5**d * Ps(args) * Ps(args), ranges)

    return res


def eval_xi_chi_st(basis, l, s, t):
    """
    Given indices l, s, t, compute the quantity
    <<xi_lchi_schi_t>>
    """

    Ps = basis[s]
    Pt = basis[t]
    d = len(Ps._index)

    ranges = [[-1, 1] for _ in range(d)]

    res, _ = integrate.nquad(lambda *args: 0.5**d * args[l] * Ps(args) * Pt(args), ranges)

    return res


def realise(u, d, p, w):
    """
    Given a solution coefficient matrix u, then dimensions used to generate it
    d, p and a value for omega w. Construct the realisation for the process for
    that particular value of omega
    """
    basis = legendre_chaos(d, p)
    P, N = u.shape

    us = np.zeros((N,))

    for j in range(N):
        for s in range(P):
            us[j] += u[s, j]*basis[s](w)

    return us

def calc_variance(polybasis, u):
    """
    Given the stochastic basis, polybasis used to create the solution
    coefficient matrix u return the variance of the process
    """

    P, N = u.shape
    var = np.zeros((N,))
    spatial_basis = hat_basis(-1, 1, N)

    for s in range(1, P):

        chi_sq = eval_chi_s_squared(polybasis, s)

        for j in range(1, N-1):

            var[j] += (u[s, j] * (u[s, j-1] + u[s, j] + u[s,j+1]))*chi_sq

    return var
