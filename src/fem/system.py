import numpy as np
from math import sqrt

from obj.generate import rectangular_domain
from .construction import construct_global_stiffness_matrix,\
                          construct_global_mass_matrix


def L2_error(grid, soln, exact):
    """
    Given the discrete grid, the approximate solution and
    the exact solution, compute the L2 error of the difference
    """

    xs, ys = grid
    err = 0

    for y in ys:
        for x in xs:
            err += (soln[x, y] - exact(x, y)) ** 2

    err = (1 / len(xs) ** 2) * sqrt(err)

    return err


def solve_system(parameters, known_nodes, f, N, exact=None):
    """
    Given the parameters and the domain which specifies the problem,
    construct and solve the finite element method system. Return a dictionary
    containing all the goodies

    Arguments:
            - parameters:  A 3-tuple specifiying the parameters of the system
            - known_nodes: A predicate function in (x,y) to determine if the
                           nodes are known
            - f:           A function in (x,y) representing the RHS of the PDE
            - N:           The resolution of the discretisation to use
            - exact:       (Optional). A function in (x,y) which specifies the
                           exact solution. If provided then extra comparisons
                           with our approximate solution will be generated.

    Returns:
            A dictionary with the following fields:

              - A:    The global stiffness matrix
              - M:    The global mass matrix
              - grid: The discretised domain of the problem
              - U:    The solution at each point in the grid
              - err:  The L^2 error if the exact solution is provided
    """

    # Everything we will return goes here
    results = {}

    # Construct the domain based on our N
    obj = rectangular_domain(N=N)
    domain = {'obj': obj, 'known_nodes': known_nodes}

    # Construct the global matrices
    A = construct_global_stiffness_matrix(domain, parameters)
    M = construct_global_mass_matrix(domain, parameters)

    # We will return the constructed matrices for debugging purposes
    results['A'] = A.A
    results['M'] = M.A

    # Construct the discretised version of the RHS
    F = np.array([f(x, y) for x, y, _ in obj['vertices']])

    # Solve the FEM system
    u = np.linalg.solve(A.A, M*F)

    # Now! The solution u only contains the values for the unknown nodes! But
    # the solution must also include the values of the known nodes, so we will
    # add these in now
    u = np.append(u, [0.0 for _ in range(4*N)])    # TODO: Make this general!

    # Due to the way we construct this system and the solution all of the nodes
    # are in the wrong order when it comes to plotting the system. So before
    # we pass it back to the user we need to tidy this up.

    # First we need to associate each point in our grid with its solution value
    xs = [x for x, _, _ in obj['vertices']]
    ys = [y for _, y, _ in obj['vertices']]
    soln = list(zip(xs, ys, u))

    # Now we need to rearrange these point so they are in a sane order
    uniq_x = sorted(list(set(xs)))
    uniq_y = sorted(list(set(ys)))

    # Then we map the xs and ys to their index in the uniq_* lists
    indcies = ((uniq_x.index(xi), uniq_y.index(yi), ui) for xi, yi, ui in soln)

    # From which we construct the solution which is now ordered correctly
    U = np.matrix(np.identity(N + 1))
    for x, y, u in indcies:
        U[x, y] = u

    # Finally! We can return the grid and the solution
    results['U'] = U
    results['grid'] = np.meshgrid(uniq_x, uniq_y)

    # Final touch, if we were given the exact solution comute the error
    if exact is not None:
        results['err'] = L2_error((uniq_x, uniq_y), U, exact)

    return results
