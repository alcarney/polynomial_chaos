import numpy as np


def local_mass_matrix(h):
    """
    This defines the local mass matrix, this is only dependant
    on the resolution of the mesh, given by h
    """

    return h**2 * np.matrix([[1/12, 1/24, 1/24],
                             [1/24, 1/12, 1/24],
                             [1/24, 1/24, 1/12]])


def local_stiffness_matrix(a, b, c, h):
    """
    This defines the local stiffness matrix for the general laplace equation.

    It takes in the scalars a, c and the vector b which characterise the system
    and h which represents the resolution of the mesh we are using and
    will return the 3x3 local stiffness matrix.
    """

    m1 = np.matrix([[1   , -1/2, -1/2],
                    [-1/2, 1/2 , 0],
                    [-1/2, 0   , 1/2]])

    m2 = np.matrix([[-(b[0] + b[1])/6, b[0]/6, b[1]/6],
                    [-(b[0] + b[1])/6, b[0]/6, b[1]/6]           ,
                    [-(b[0] + b[1])/6, b[0]/6, b[1]/6]])

    m3 = np.matrix([[1/12, 1/24, 1/24],
                    [1/24, 1/12, 1/24],
                    [1/24, 1/24, 1/12]])

    return (a * m1) + (h * m2) + (c * h**2 * m3)

