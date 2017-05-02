import numpy as np
import scipy.integrate as integrate

from math import sqrt
from scipy.sparse import bmat, csr_matrix, coo_matrix
from scipy.sparse.linalg import spsolve

from fem.twod_deterministic import local_mass_matrix
from fem.polynomial_chaos import legendre_chaos, eval_chi_s_squared,\
                                 eval_xi_chi_st, read_eigendata


def local_diagonal_stiffness_matrix(eps, mu, basis, s):
    """
    Given the mean of the random process mu, the constant eps the index
    of the stochastic basis s and the basis itself, return
    the local stiffness matrix for the particular block diagonal matrix
    """

    # Evaluate <<chi_s^2>>
    chi = eval_chi_s_squared(basis, s)

    A = np.array([[2, -1, -1], [-1, 1, 0], [-1, 0, 1]])

    return (((1 + eps*mu) * chi) / 2) * A


def assemble_stiffness_matrix(N, Ak):
    """
    Given N, representing the density of the mesh, and Ak, representing the
    local stiffness matrix - assemble the 'global' stiffness matrix
    """

    # Some convenient definitions
    L = N - 1
    cols = [] ; rows = [] ; values = []

    # Loop through each interior node - be careful of the boundary!
    for i in range(L):
        for j in range(L):

            # Compute the index k
            k = i*L + j

            # Add the non zero entries to row k of A - checking that the nodes
            # don't lie on the boundary!
            rows.append(k); cols.append(k); values.append(2*(Ak[0,0] + Ak[1,1] + Ak[2,2]))

            if (j + 1) <= (L - 1):
                rows.append(k); cols.append(i*L+(j+1)); values.append(Ak[0, 1] + Ak[1, 0])

            if (j - 1) >= 0:
                rows.append(k); cols.append(i*L + (j-1)); values.append(Ak[1, 0] + Ak[0, 1])

            if i - 1 >= 0:
                rows.append(k); cols.append((i-1)*L + j); values.append(Ak[0, 2] + Ak[2, 0])

            if i + 1 <= (L - 1):
                rows.append(k); cols.append((i+1)*L + j); values.append(Ak[2, 0] + Ak[0, 2])

            if j + 1 < (L - 1) and i - 1 >= 0:
                rows.append(k); cols.append((i-1)*L + (j+1)); values.append(Ak[1, 2] + Ak[2, 1])

            if j - 1 >= 0 and i + 1 <= (L - 1):
                rows.append(k); cols.append((i+1)*L + (j-1)); values.append(Ak[2, 1] + Ak[1, 2])

    # Construct A in sparse format
    A = coo_matrix((values, (rows, cols)), shape=(L**2, L**2)).tocsr()

    return A


def construct_diagonal_stiffness_matrix(N, eps, mu, basis, s):
    """
    Given N representing the density of the mesh, parameters mu, eps, the
    stochastic basis and index s construct the diagonal block stiffness matrix
    """

    # Get the local stiffness matrix
    Ak = local_diagonal_stiffness_matrix(eps, mu, basis, s)

    # Assemble A
    A = assemble_stiffness_matrix(N, Ak)

    return A


def construct_mass_matrix(N, basis):
    """
    Given N representing the density of the mesh and the stochastic construct the
    global mass matrix for the system
    """

    # Calculate h
    h = 1/N

    # Some convenient definitions - note P in the case has nothing to do with the
    # stochastic basis, I'm just not very good at new names
    L, P = (N - 1), (N + 1)

    # Get the local mass matrix
    Mk = local_mass_matrix(h)

    # Construct M_1
    rows, cols, values = [], [], []

    for i in range(L):
        for j in range(L):

            # Compute k
            k = i*L + j

            # Add the non zero entries of row k of M
            rows.append(k); cols.append((i+1)*P+(j+1)); values.append(2*(Mk[0,0] + Mk[1,1] + Mk[2,2]))
            rows.append(k); cols.append((i+1)*P+(j+2)); values.append(Mk[0, 1] + Mk[1, 0])
            rows.append(k); cols.append((i+1)*P + j); values.append(Mk[1, 0] + Mk[0, 1])
            rows.append(k); cols.append(i*P + (j+1)); values.append(Mk[0, 2] + Mk[2, 0])
            rows.append(k); cols.append((i+2)*P + (j+1)); values.append(Mk[2, 0] + Mk[0, 2])
            rows.append(k); cols.append(i*P + (j+2)); values.append(Mk[1, 2] + Mk[2, 1])
            rows.append(k); cols.append((i+2)*P + j); values.append(Mk[2, 1] + Mk[1, 2])

            M1 = coo_matrix((values, (rows, cols)), shape=(L**2, P**2)).tocsr()

    # With M1 constructed, now just pad M with enough zeros
    Z = csr_matrix(np.zeros((L**2, P**2)))
    M = [[Z] for _ in range(len(basis))]
    M[0][0] = M1

    return bmat(M).tocsc()
    return M1
