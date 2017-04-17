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


def local_mass_matrix(h, s):
    """
    Given the number of the legendre polynomial basis s, and h
    representing the length of the interval return the corresponding
    local mass matrix.
    """

    M = np.array([[2, 1], [1, 2]])

    return (h/3*(2*s + 1))*M


def assemble_diagonal_stiffness_matrix(N, mu, s):
    """
    Assemble the sth-sth matrix in the global stiffness matrix
    for given mean mu, resolution N and legendre polynomial index s
    """

    # Calculate h
    h = 1/N

    # Get the local stiffness matrix
    A_k = local_diagonal_stiffness_matrix(mu, h, s)

    # Construct the 'global' matrix
    A = np.zeros((N - 1, N - 1))
    i, j = np.indices(A.shape)

    A[i == j + 1] = A_k[1, 0]               # Subdiagonal
    A[i == j] = A_k[1, 1] + A_k[0, 0]       # Main diagonal
    A[i == j - 1] = A_k[0, 1]               # Superdiagonal

    return A


def assemble_mass_matrix(N, s):
    """
    Assemble the sth-sth matrix in the global mass matrix
    for given reolution N and legendre polynomial index s
    """

    # Calculate h
    h = 1/N

    # Get the local mass matrix
    M_k = local_mass_matrix(h, s)

    # Get the 'global' mass matrix
    M = np.zeros((N - 1, N + 1))
    i, j = np.indices(M.shape)

    M[i == j] = M_k[1, 0]                   # Subdiagonal
    M[i == j - 1] = M_k[1, 1] + M_k[0, 0]   # Main diagonal
    M[i == j - 2] = M_k[0, 1]               # Superdiagonal

    return M
