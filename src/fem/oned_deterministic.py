import numpy as np
from math import sqrt


"""
This file contains all the code necessary to run the finite element method
in the one dimensional deterministic case of Laplace's Equation, in the
unit interval with zero boundary condition:

    - au''(x) + bu'(x) + cu(x) = f(x),     x in [0,1]
    - u(0) = 0 = u(1)
"""


def local_stiffness_matrix(a, b, c, h):
    """
    Given the parameters a,b,c which define the problem above
    and h which represents the length of the subintervals.
    Construct the corresponding local stiffness matrix
    """

    A1 = np.array([[1, -1], [-1, 1]])
    A2 = np.array([[-1, 1], [-1, 1]])
    A3 = np.array([[2, 1], [1, 2]])

    return (a/h)*A1 + (b/2)*A2 + (c*h/6)*A3


def local_mass_matrix(h):
    """
    Given h representing the length of the subintervals,
    construct the corresponding local mass matrix
    """

    return (h/6) * np.array([[2, 1], [1, 2]])


def construct_system(N=4, a=1, b=0, c=0):
    """
    Construct the global mass and stiffness matrices
    for the given problem

    Arguments:
        N: An integer. Represents the density of the discretisation.
           Default value: 4

        a: A float. Corresponds to the parameter a in the equation
           Default value: 1

        b: A float. Corresponds to the parameter b in the equation
           Default value: 0

        c: A float. Corresponds to the parameter c in the equation
           Default value: 0

    Returns:
        A: The global stiffness matrix, dimension N x N
        M: The global mass matrix, dimension N x (N + 1)
    """

    # Calculate h
    h = 1/N

    # Get the local mass and stiffness matrices for the given problem
    A_k = local_stiffness_matrix(a, b, c, h)
    M_k = local_mass_matrix(h)

    # Construct the global stiffness matrix
    A = np.zeros((N - 1, N - 1))
    i, j = np.indices(A.shape)      # We use these to access the diagonals of A

    A[i == j + 1] = A_k[1, 0]          # Subdiagonal
    A[i == j] = A_k[1, 1] + A_k[0, 0]  # Main diagonal
    A[i == j - 1] = A_k[0, 1]          # Superdiagonal

    # Construct the global mass matrix
    M = np.zeros((N - 1, N + 1))
    i, j = np.indices(M.shape)      # We use these to access the diagonals of M

    M[i == j] = M_k[1, 0]                   # Subdiagonal
    M[i == j - 1] = M_k[1, 1] + M_k[0, 0]   # Main diagonal
    M[i == j - 2] = M_k[0, 1]               # Superdiagonal

    return A, M


def solve_system(f, N=4, a=1, b=0, c=0):
    """
    Given the density of the nodes N and a function f representing
    the RHS, assemble the linear system Au = Mf and solve for u
    """

    # Discretise the domain
    xs = np.linspace(0, 1, N + 1)

    # Approximate f
    fs = np.array([f(x) for x in xs])

    # Construct the global matrices for the problem
    A, M = construct_system(N, a, b, c)

    # Construct the RHS of Au = Mf
    Mf = np.dot(M, fs)

    # Solve the system, and add on the boundary points
    us = np.linalg.solve(A, Mf)
    U = [0] + us.tolist() + [0]

    # Solve and return subdivided domain and solution
    return xs, U


def L2_error(u, U, N):
    """
    Given a function u representing the exact solution for the problem
    an array U representing our approximate solution to the problem
    and N representing the density of the nodes, return the L2 norm
    of the error ||u - U||_2
    """

    # Initially assume no error and recreate the domain
    err = 0
    xs, h = np.linspace(0, 1, N + 1, retstep=True)

    # Add the error at each point
    for i in range(len(U)):
        err += (u(xs[i]) - U[i])**2

    # Scale by the length of the subintervals and, square root
    err = sqrt(h*err)

    return err


def hat_basis_fn(xkm, xk, xkp):


    def phi_k(x):

        if x < xkm or x > xkp:
            return 0

        if x < xk:
            return (x - xkm)/(xk - xkm)
        else:
            return (xkp - x)/(xkp - xk)

    return phi_k

def hat_basis(a, b, N):

    xs, h = np.linspace(a, b, N+1, retstep=True)

    # A bit of care is needed for x_0
    bases = [hat_basis_fn(a-h, a, a+h)]

    # Build the middle
    for k in range(1, N):
        bases.append(hat_basis_fn(xs[k-1], xs[k], xs[k+1]))

    # And some care is needed for x_N
    bases.append(hat_basis_fn(b-h, b, b+h))

    return bases
