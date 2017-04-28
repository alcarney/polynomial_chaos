import numpy as np
from math import sqrt
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


def local_stiffness_matrix(a, b, c, h):
    """
    Given scalars a, c and 1x2 vector b along with h representing
    the size of the triangular elements, construct the local
    stiffness matrix for the problem
    """

    m1 = np.array([[ 2, -1, -1],
                   [-1,  1,  0],
                   [-1,  0,  1]])

    m2 = np.array([[-(b[0] + b[1])/6, b[0]/6, b[1]/6],
                   [-(b[0] + b[1])/6, b[0]/6, b[1]/6],
                   [-(b[0] + b[1])/6, b[0]/6, b[1]/6]])

    m3 = np.array([[2, 1, 1],
                   [1, 2, 1],
                   [1, 1, 2]])

    return (a/2)*m1 + (h**2 * m2) + ((c*h**2)/24)*m3


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

    # Calculate h, L, M
    h = 1/N
    L = (N - 1)
    P = (N + 1)

    # Get the local stiffness and mass matrices
    Ak = local_stiffness_matrix(a, b, c, h)
    Mk = local_mass_matrix(h)

    Ac = []; Ar = []; Av = []
    Mc = []; Mr = []; Mv = []

    # Loop through each interior node, being careful of hitting boundary nodes
    for i in range(L):
        for j in range(L):

            # Compute k
            k = i*L + j

            # Add the non zero entries of row k of M
            Mr.append(k); Mc.append((i+1)*P+(j+1)); Mv.append(2*(Mk[0,0] + Mk[1,1] + Mk[2,2]))
            Mr.append(k); Mc.append((i+1)*P+(j+2)); Mv.append(Mk[0, 1] + Mk[1, 0])
            Mr.append(k); Mc.append((i+1)*P + j); Mv.append(Mk[1, 0] + Mk[0, 1])
            Mr.append(k); Mc.append(i*P + (j+1)); Mv.append(Mk[0, 2] + Mk[2, 0])
            Mr.append(k); Mc.append((i+2)*P + (j+1)); Mv.append(Mk[2, 0] + Mk[0, 2])
            Mr.append(k); Mc.append(i*P + (j+2)); Mv.append(Mk[1, 2] + Mk[2, 1])
            Mr.append(k); Mc.append((i+2)*P + j); Mv.append(Mk[2, 1] + Mk[1, 2])

            # Add the non zero entries of row k of A - checking of course that the given
            # entries make sense
            Ar.append(k); Ac.append(k); Av.append(2*(Ak[0, 0] + Ak[1, 1] + Ak[2, 2]))

            if (j + 1) <= (L - 1):
                Ar.append(k); Ac.append(i*L+(j+1)); Av.append(Ak[0, 1] + Ak[1, 0])

            if (j - 1) >= 0:
                Ar.append(k); Ac.append(i*L + (j-1)); Av.append(Ak[1, 0] + Ak[0, 1])

            if i - 1 >= 0:
                Ar.append(k); Ac.append((i-1)*L + j); Av.append(Ak[0, 2] + Ak[2, 0])

            if i + 1 <= (L - 1):
                Ar.append(k); Ac.append((i+1)*L + j); Av.append(Ak[2, 0] + Ak[0, 2])

            if j + 1 < (L - 1) and i - 1 >= 0:
                Ar.append(k); Ac.append((i-1)*L + (j+1)); Av.append(Ak[1, 2] + Ak[2, 1])

            if j - 1 >= 0 and i + 1 <= (L - 1):
                Ar.append(k); Ac.append((i+1)*L + (j-1)); Av.append(Ak[2, 1] + Ak[1, 2])

    # Convert the lil_matrix to csr_matrix for quicker solving
    A = coo_matrix((Av, (Ar, Ac)), shape=(L**2,L**2)).tocsr()
    M = coo_matrix((Mv, (Mr, Mc)), shape=(L**2,P**2)).tocsr()

    return A, M


def solve_system(f, N, a, b, c):
    """
    Given the density of the mesh N and function f representing the RHS
    along with the coefficients a, b, c assemble the linear system Au = Mf and
    solve for u
    """

    # Discretise the domain
    xs = np.linspace(0, 1, N + 1)
    ys = np.linspace(0, 1, N + 1)

    # Approximate f
    fs = np.array([f(x,y) for x in xs for y in ys])

    # Construct the global matrices for the problem
    A, M = construct_system(N, a, b, c)

    # Construct the RHS of Au = Mf
    Mf = M*fs

    # Solve the system
    us = spsolve(A, Mf)

    # Add in the boundary conditions
    us = us.reshape((N - 1, N - 1))
    U = np.zeros(tuple(n + 2 for n in us.shape))
    U[tuple(slice(1, -1) for n in us.shape)] = us

    return xs, ys, U


def L2_error(u, U, N):
    """
    Given a function u representing the exact solution to the problem, an array
    U representing our approximate solution to the problem and N representing
    the density of the mesh, return the L2 norm of the error || u - U ||_2
    """

    # Initially assume zero error and recreate the domain
    err = 0
    xs, h = np.linspace(0, 1, N+1, retstep=True)
    ys = np.linspace(0, 1, N+1)

    # Add the error at each point
    for i in range(len(xs)):
        for j in range(len(ys)):
            err += (u(xs[i], ys[j]) - U[i, j])**2

    # Scale by the size of the triangles and square root
    err = sqrt((h**2)*err)

    return err
