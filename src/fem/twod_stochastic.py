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


def local_off_diagonal_stiffness_matrix(h, beta):
    """
    Given the eigenfunction beta, and h representing the scale
    of the element construct the correspinding local
    stiffness matrix
    """

    # Define grad of the basis functions for the reference triangle
    grad_phi_1 = 1/h * np.array([-1, -1])
    grad_phi_2 = 1/h * np.array([1, 0])
    grad_phi_3 = 1/h * np.array([0, 1])

    basis = [grad_phi_1, grad_phi_2, grad_phi_3]

    # Define the grad . grad part of the integrand
    def grad_dot_grad(pm, pn):
        return (pm[0] * pn[0]) + (pm[1] * pn[1])

    # The matrix is 3x3
    Ak = np.zeros((3,3))

    for i in range(3):
        for j in range(3):

            # Do the integration
            Ak[i, j] = h**2 * integrate.dblquad(lambda y, x: beta(x, y) * grad_dot_grad(basis[i], basis[j]),
                                                0, 1, lambda x: 0, lambda x: 1 - x)[0]

    return Ak



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


def construct_off_diagonal_stiffness_matrix(N, basis, l, s, t, eps, lmda, beta):
    """
    Given N representing the density of the mesh, the stochasic basis, s and t
    indices into the basis, l reprenting thel-th stochastick variable, epsilon and
    eigenvalue/function lmda, beta construct the off diagonal matrix
    """

    # Calculate h
    h = 1/N

    # Get the local matrix for the given beta
    Ak = local_off_diagonal_stiffness_matrix(h, beta)

    # Construct the matrix
    A = assemble_stiffness_matrix(N, Ak)

    # Compute <<xi_lchi_schi_t>>
    xi_chi = eval_xi_chi_st(basis, l, s, t)

    return (3 * eps * sqrt(lmda) * xi_chi) * A


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


def construct_system(N, basis, d, p, eps, mu):
    """
    Given N representing the density of the mesh, d, p reprsenting the dimensions
    of the stochastic space and parameters eps, mu, construct and return the
    global stiffness and mass matrices for the system
    """

    P = len(basis)

    # Read the eigendata
    eigendata = read_eigendata('expansion-data.csv')
    ls = [val['lambda'] for val in eigendata]
    bs = [val['beta'] for val in eigendata]

    # Don't forget! Now that we are in 2D the eigen values/functions are given by pairs
    # of the 1D results!
    idx = [(i, j) for i in range(len(ls)) for j in range(len(ls))]

    # But the above would give (0, 0), (0,1), (0,2) ... etc. so we'll sort these by their
    # sum so the sequence becomes (0,0), (0,1) (1,0) (1,1) ...
    indices = [(i,j) for i, j in sorted(idx, key=sum)]

    # Now assemble the pairs
    lambdas = [ls[i] * ls[j] for i, j in indices]
    betas = [lambda x, y: bs[i](x) * bs[j](y) for i, j in indices]

    # Construct the block form of A
    Ast = [[None for s in range(P)] for t in range(P)]

    for s in range(P):
        for t in range(P):

            # Diagonals
            if s == t:
                mat = construct_diagonal_stiffness_matrix(N, eps, mu, basis, s)
                Ast[s][t] = mat
            else:
                mat = sum([construct_off_diagonal_stiffness_matrix
                            (N, basis, l, s, t, eps, lambdas[l], betas[l]) for l in range(d)])
                Ast[s][t] = mat

    A = bmat(Ast).tocsc()
    M = construct_mass_matrix(N, basis)


    return A, M


def solve_system(N, d, p, eps, mu, f):
    """
    Given N representing the density of the mesh, d, p representing the
    dimension of the stochastic space, parameters eps, mu  and right hand side f
    construct and solve the stochastic system
    """

    basis = legendre_chaos(d, p)
    P = len(basis)

    # Construct the matrices
    A, M = construct_system(N, basis, d, p, eps, mu)

    # Approximate f
    xs = np.linspace(-1, 1, N+1)
    ys = np.linspace(-1, 1, N+1)
    fs = np.array([f(x, y) for x in xs for y in ys])

    # Construct the RHS
    Mf = M * fs

    # Solve
    us = spsolve(A, Mf)

    # us is now an array of matrices, each matrix represents the contributions
    # from each of the stochastic basis vectors
    us = us.reshape((P, N - 1, N - 1))

    # Add the boundary nodes (all zero)
    U = np.zeros((P, N + 1, N + 1))

    for s in range(P):
        U[s][slice(1, -1), slice(1,-1)] = us[s]

    return xs, ys, U


def calc_variance(basis, U):
    """
    Given the stochastic basis and the solution proces U, calcluate the
    variance
    """
    P, N, _ = U.shape
    var = np.zeros((P, N, N))

    for s in range(1,P):

        chi_sq = eval_chi_s_squared(basis, s)

        # Loop over each point in space, now normally we would have to check to
        # make sure that we didn't consider points outside the boundary but as
        # the problems we consider are all zero on the boundary we can get away with
        # looping over the interior nodes as they are the ony non zero values anyway
        for i in range(1, N-1):
            for j in range(1, N-1):

                # Add all the contributions to the variance
                var[s][i,j]  += U[s, i, j] * U[s, i, j]
                var[s][i, j] += U[s, i, j] * U[s, i, j+1]
                var[s][i, j] += U[s, i, j] * U[s, i, j-1]
                var[s][i, j] += U[s, i, j] * U[s, i+1, j]
                var[s][i, j] += U[s, i, j] * U[s, i-1, j]
                var[s][i, j] += U[s, i, j] * U[s, i+1, j-1]
                var[s][i, j] += U[s, i, j] * U[s, i-1, j+1]

                # Don't forget to include <<chi_s^2>>
                var[s][i, j] *= chi_sq

    # Finally sum up along the probability axis
    return np.sum(var, axis=0)

