import numpy as np
import scipy.integrate as integrate

from scipy.sparse import bmat, csr_matrix
from scipy.sparse.linalg import spsolve
from fem.oned_deterministic import local_mass_matrix
from fem.polynomial_chaos import legendre_chaos, eval_chi_s_squared,\
                                eval_xi_chi_st, read_eigendata


def local_diagonal_stiffness_matrix(mu, basis, s, h):
    """
    Given the mean of the random process mu, the length of the
    interval h and the number of the legendre polynomial basis
    s, this function returns the appropriate local sitffness
    matrix for a matrix on the digaonal of the global system.
    """

    # Calculate <<\chi_s^2>>
    chi = eval_chi_s_squared(basis, s)

    A = np.array([[1, -1], [-1, 1]])

    return (mu*chi/h)*A


def local_off_diag_stiffness_matrix(xk, xk1, beta):
    """
    Given the endpoints of the interval xk, xk1 and the eigenfunction
    beta construct the local stiffness matrix
    """

    # Calculate the length of the inteval
    h = xk1 - xk

    # The matrix will be 2x2
    Ak = np.zeros((2, 2))

    # Build each value for the matrix
    for i in range(2):
        for j in range(2):

            # Since we are takin h out the basis functions become +/- 1 this
            # figure sout the sign the integrand will have
            sign = ((-1)**i) * ((-1)**j)

            Ak[i, j] = (1/h**2) * integrate.quad(lambda x: sign*beta(x), xk, xk1)[0]

    return Ak


def assemble_mass_matrix(N, P):
    """
    Given N and P representing resolution of the physical and stoachastic
    discretisations respectively, assemble the global mass matrix
    """

    # Calculate h and Mk
    h = 1/N
    Mk = local_mass_matrix(h)

    # Construct M_0
    Z = np.zeros((N - 1, N + 1))
    M0, Z = Z, csr_matrix(Z)
    i, j = np.indices(M0.shape)      # We use these to access the diagonals of M

    M0[i == j] = Mk[1, 0]                  # Subdiagonal
    M0[i == j - 1] = Mk[1, 1] + Mk[0, 0]   # Main diagonal
    M0[i == j - 2] = Mk[0, 1]              # Superdiagonal

    # Assemble M
    M = [[Z] for _ in range(P)]
    M[0][0] = M0

    return bmat(M).tocsc()


def assemble_diagonal_stiffness_matrix(N, mu, basis, s):
    """
    Assemble the sth-sth matrix in the global stiffness matrix
    for given mean mu, resolution N and legendre polynomial index s
    """

    # Calculate h
    h = 1/N

    # Get the local stiffness matrix
    A_k = local_diagonal_stiffness_matrix(mu, basis, s, h)

    # Construct the 'global' matrix
    A = np.zeros((N - 1, N - 1))
    i, j = np.indices(A.shape)

    A[i == j + 1] = A_k[1, 0]               # Subdiagonal
    A[i == j] = A_k[1, 1] + A_k[0, 0]       # Main diagonal
    A[i == j - 1] = A_k[0, 1]               # Superdiagonal

    return A


def assemble_off_diagonal_stiffness_matrix(N, basis, l, s, t, lmda, beta):
    """
    Assemble the sth-tth matrix in the global stiffness matrix
    for given resolution N, legendre polynomial indices s and t
    and eigenvalue/eigenfunction pair lmda, beta
    """

    # Discreteise the domain - we will need the xis
    xs = np.linspace(-1, 1, N + 1)

    # Evaluate n<<\xi_l\chi_s\chi_t>>
    coef = lmda * eval_xi_chi_st(basis, l, s, t)

    # Construct the 'global' matrix
    A = np.zeros((N - 1, N - 1))
    i, j = np.indices(A.shape)

    diagonal = []
    superdiagonal = []
    subdiagonal = []

    # Do this first entry, indices are hard!
    A0 = local_off_diag_stiffness_matrix(xs[0], xs[1], beta)
    Ak = local_off_diag_stiffness_matrix(xs[1], xs[2], beta)
    diagonal.append(A0[1, 1] + Ak[0, 0])

    # Assemble the 'global' matrix - taking care to make sure we
    # take into account the changing local matrix
    for k in range(1, N - 1):

        # Get the local matrix for the current interval
        Anext = local_off_diag_stiffness_matrix(xs[k+1], xs[k+2], beta)

        # Add the appropraite entries to the diagonals
        diagonal.append(Ak[1, 1] + Anext[0, 0])
        superdiagonal.append(Ak[0, 1])
        subdiagonal.append(Ak[1, 0])

        # Anext will be Ak in the next loop
        Ak = Anext

    A[i == j] = np.array(diagonal)
    A[i == j - 1] = np.array(superdiagonal)
    A[i == j + 1] = np.array(subdiagonal)

    return coef * A


def construct_global_stiffness_matrix(N, basis, d, p, mu):
    """
    Assemble the global stiffness matrix
    """
    P = len(basis)

    eigendata = read_eigendata('expansion-data.csv')
    lambdas = [val['lambda'] for val in eigendata]
    betas = [val['beta'] for val in eigendata]

    Ast = [[None for s in range(P)] for t in range(P)]

    for s in range(P):
        for t in range(P):

            # Is this matrix on the diagonal
            if s == t:
                mat = assemble_diagonal_stiffness_matrix(N, mu, basis, s)
                Ast[s][t] = mat
            else:
                mat = sum([assemble_off_diagonal_stiffness_matrix
                            (N, basis, l, s, t, lambdas[l], betas[l]) for l in range(d)])
                Ast[s][t] = csr_matrix(mat)

    # Assemble the block matrices into the global matrix
    A = bmat(Ast)

    return A.tocsc()


def solve_system(N, d, p, mu, f):
    """
    Construct the global stiffness and mass matrices for the given parameters
    """

    # Get the basis for the stochastic system
    basis = legendre_chaos(d, p)
    P = len(basis)
    print(P)

    # Approximate f
    xs = np.linspace(-1, 1, N+1)
    F = np.array([f(x) for x in xs])

    # Assemble the linear system
    A = construct_global_stiffness_matrix(N, basis, d, p, mu)
    M = assemble_mass_matrix(N, P)

    # Solve
    return spsolve(A, M*F)
