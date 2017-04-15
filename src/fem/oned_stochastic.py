import numpy as np


def local_diagonal_stiffness_matrix(mu, h, s):
    """
    Given the mean of the random process mu, the length of the
    interval h and the number of the legendre polynomial basis
    s, this function returns the appropriate local sitffness
    matrix for a matrix on the digaonal of the global system.
    """

    A = np.array([[1, -1], [-1, 1]])

    return ((2*mu)/(h*(2*s + 1)))*A



