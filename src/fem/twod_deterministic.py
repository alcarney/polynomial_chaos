import numpy as np


def local_stiffness_matrix(a, b, c, h):
    """
    Given scalars a, c and 1x2 vector b along with h representing
    the size of the triangular elements, construct the local
    stiffness matrix for the problem
    """

    m1 = np.array([[ 2, -1, -1],
                   [-1,  1,  0],
                   [-1,  0,  1]])

    m2 = np.array([[-(b[0] + b[1]), 2*b[0], 2*b[1]],
                   [-(b[0] + b[1]), 2*b[0], 2*b[1]],
                   [-(b[0] + b[1]), 2*b[0], 2*b[1]]])

    m3 = np.array([[2, 1, 1],
                   [1, 2, 1],
                   [1, 1, 2]])

    return (a/2)*m1 + (h/12)*m2 + ((c*h**2)/24)*m3


def local_mass_matrix(h):
    """
    Given h, which represents the size of the triangular element
    construct the local mass matrix
    """

    return (h**2/24) * np.array([[2, 1, 1],
                                 [1, 2, 1],
                                 [1, 1, 2]])


def construct_system(N, a, b, c):
    """
    Given N, representing the density of the mesh along with the scalars
    a, c and 1x2 vector b which define the problem construct the global
    matrices A and M
    """

    # Calculate h
    h = 1/N

    # Get the local stiffness and mass matrices
    A_k = local_stiffness_matrix(a, b, c, h)
    M_k = local_mass_matrix(h)

    # Consruct the global stiffness matrix
    A = np.zeros((N - 1, N - 1))
    i, j = np.indcies(A.shape)


    # Construct the global mass matrix
    M = np.zeros(N - 1, N + 1)
    i, j = np.indcies(M.shape)

    return A, M

