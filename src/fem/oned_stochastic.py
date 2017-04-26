import numpy as np
import scipy.integrate as integrate

from scipy.sparse import bmat, csr_matrix
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


def local_mass_matrix(h, basis, s):
    """
    Given the number of the legendre polynomial basis s, and h
    representing the length of the interval return the corresponding
    local mass matrix.
    """
    chi = eval_chi_s_squared(basis, s)

    M = np.array([[2, 1], [1, 2]])

    return (h*chi/6)*M


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


def construct_global_stiffness_matrix(N, d, p, mu):
    """
    Assemble the global stiffness matrix
    """
    basis = legendre_chaos(d, p)
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
    A = bmat(Ast).tocsr()

    return A
