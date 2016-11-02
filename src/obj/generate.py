import numpy as np
import logging

logging.basicConfig(format="%(name)s :: [%(levelname)s]: %(message)s", level=logging.DEBUG)
logger = logging.getLogger(__name__)


def rectangular_domain(x_min=0, x_max=1, y_min=0, y_max=1, N=4):
    """
    This function generates the rectangular mesh
    (x_min, x_max) x (y_min, y_max) with a regular triangulation.

    It also puts the points in "the order" which we apparently need
    for things to work

    Depending on how complex this becomes computationally, it might be
    worth writing an obj exporter also...

    Arguments:
        - x_min: A float. The minimum x value to start at
        - x_max: A float. The maximum x value to finish at
        - y_min: A float. The minimum y value to start at
        - y_max: A float. The maximum y value to finish at
        - N:     An integer. Represents the resolution of the mesh we want.

    Returns:
        A dictionary with the fields:

            - vertices: A list of 3-tuples containing the x,y,z coords
                        of points in the mesh
            - faces:    A list of 3-tuples containing the indices into the
                        list above of the points which comprise a face
            - h:        Representing the step size in the resolution of the
                        mesh
    """

    logger.info("Generating triangular mesh for domain: [%.2f, %.2f] x [%.2f, %.2f]" %
                (x_min, x_max, y_min, y_max))

    # The dimension of the interior nodes
    M = N - 1

    # Generate the x and y points at the resolution requested
    # At the moment we will assume that h_x and h_y will be the same
    xs, h_x = np.linspace(x_min, x_max, num=N+1, endpoint=True, retstep=True)
    ys, _ = np.linspace(y_min, y_max, num=N+1, endpoint=True, retstep=True)

    logger.info("N = %i => h = %.6f" % (N, h_x))

    # Pull out the boundary x and y values
    boundary_x = np.array([xs[0], xs[-1]])
    boundary_y = np.array([ys[0], ys[-1]])

    # And the interior nodes
    interior_x = xs[1:-1]
    interior_y = ys[1:-1]

    # Now we construct the boundary nodes
    # {x_min, x_max} X [y_min, y_max] U {y_min, y_max} X [x_min, x_max]
    boundary_nodes = [np.array([boundary_x[0], boundary_y[0], 0])] +\
                     [np.array([x, boundary_y[0], 0]) for x in interior_x] +\
                     \
                     [np.array([boundary_x[1], boundary_y[0], 0])] +\
                     [np.array([boundary_x[1], y, 0]) for y in interior_y] +\
                     \
                     [np.array([boundary_x[1], boundary_y[1], 0])] +\
                     [np.array([x, boundary_y[1], 0]) for x in reversed(interior_x)] +\
                     \
                     [np.array([boundary_x[0], boundary_y[1], 0])] +\
                     [np.array([boundary_x[0], y, 0]) for y in reversed(interior_y)]

    logger.info("# of boundary nodes: %i" % len(boundary_nodes))
    logger.debug("Boundary nodes:")

    for node in boundary_nodes:
        logger.debug("  => %s" % node)

    # Next, we construct in the interior nodes
    interior_nodes = [np.array([x, y, 0]) for x in interior_x for y in interior_y]

    logger.info("# of interior nodes: %i" % len(interior_nodes))
    logger.debug("Interior nodes:")

    for node in interior_nodes:
        logger.debug("  => %s" % node)

    # Now define the vertices in the mesh
    vertices = interior_nodes + boundary_nodes

    faces = []

    # Now we need to write down the triangles, as with the vertices
    # we will start with the interior nodes. We can effectively run through a grid
    # (N-2)^2 and compute the indices as we go

    for i_x in range(1, M):
        for i_y in range(1, M):

            # First calculate the 'bottom right corner index'
            index = i_x + (i_y - 1) * M

            # Now there are two 'types' of triangle
            type1 = (index, index + M, index + 1)
            type2 = (index + N, index + 1, index + M)

            # Add them to the faces array
            faces += [type1, type2]

    # As a last step, we have to write down the triangles for the boundary nodes.
    # This will not! be fun as we have 8 cases to consider!
    # These values may come in useful
    last_int_node = M ** 2
    first_ext_node = last_int_node + 1
    last_ext_node = (N + 1) ** 2
    bottom_right_int_node = M * (M - 1) + 1
    bottom_right_ext_node = first_ext_node + N
    top_right_ext_node = first_ext_node + (2 * N)
    top_left_ext_node = first_ext_node + (3 * N)

    # First case: The bottom left corner
    type1 = (first_ext_node, first_ext_node + 1, last_ext_node)
    type2 = (1, last_ext_node, first_ext_node + 1)
    faces += [type1, type2]

    # Second case: Bottom edge run:
    for idx in range(1, M):
        index = first_ext_node + idx
        int_node = 1 + ((idx - 1) * M)

        type1 = (index, index + 1, int_node)
        type2 = (int_node + M, int_node, index + 1)

        faces += [type1, type2]

    # Third case: The bottom right corner
    type1 = (first_ext_node + M, bottom_right_ext_node, bottom_right_int_node)
    type2 = (first_ext_node + M + 2, bottom_right_int_node, bottom_right_ext_node)
    faces += [type1, type2]

    # Fourth case: Right edge run
    for idx in range(1, M):
        int_index = bottom_right_int_node + (idx - 1)
        ext_index = bottom_right_ext_node + idx

        type1 = (int_index, ext_index, int_index + 1)
        type2 = (ext_index + 1, int_index + 1, ext_index)

        faces += [type1, type2]

    # Fifth case: The top right corner
    type1 = (last_int_node, top_right_ext_node - 1, top_right_ext_node + 1)
    type2 = (top_right_ext_node, top_right_ext_node + 1, top_right_ext_node - 1)
    faces += [type1, type2]

    # Sixth Case: Top edge run
    for idx in range(1, M):
        int_index = (M - idx) * M
        ext_index = top_right_ext_node + idx

        type1 = (int_index, int_index + M, ext_index + 1)
        type2 = (ext_index, ext_index + 1, int_index + M)

        faces += [type1, type2]

    # Seventh Case: The top left corner
    type1 = (top_left_ext_node + 1, M, top_left_ext_node)
    type2 = (top_left_ext_node - 1, top_left_ext_node, M)
    faces += [type1, type2]

    # Eight case!: Left edge run - nearly done!!
    for idx in range(1, M):
        int_index = M - (idx - 1)
        ext_index = top_left_ext_node + 1 + idx

        type1 = (ext_index, int_index - 1, ext_index - 1)
        type2 = (int_index, ext_index - 1, int_index - 1)

        faces += [type1, type2]


    # As one last step, we need to shift all the indices down by 1 so they
    # match python list indexing
    faces = [(i - 1, j - 1, k - 1) for i, j, k in faces]

    logger.debug("Faces:")

    for f in faces:
        logger.debug("  => %s" % str(f))


    return {'vertices': vertices, 'faces': faces, 'h': h_x}


rectangular_domain()

