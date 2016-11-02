from scipy.sparse import coo_matrix

import numpy as np
import logging

from fem.definitions import local_mass_matrix, local_stiffness_matrix

logger = logging.getLogger(__name__)


def construct_global_mass_matrix(domain, paramters):
    """
    Given the domain of the problem and the paramters which
    characterise the system this function will construct the
    global mass matrix which we can then use to solve the
    system.

    Arugments:
        - domain: The parsed obj representing the domain
        - parameters: A tuple containing the parameters a,b,c which
                      characterise the system
    """

    obj = domain['obj']
    known_node = domain['known_nodes']

    # Calculate the value of h for our mesh
    f = obj['faces'][0]
    h = abs((obj['vertices'][f[0]] - obj['vertices'][f[1]])[0])

    # And get the corresponding local mass matrix
    local_mass = local_mass_matrix(h).A

    logger.info(" => Local Mass Matrix given by: \n%s" %
                (str(local_mass)))

    # Next create empty arrays for the row,column information
    rows = np.array([], dtype=int)
    cols = np.array([], dtype=int)
    values = np.array([])

    # Next we will need to keep track of the unknown nodes
    unknowns = []
    indices = []

    for idx, vert in enumerate(obj['vertices']):
        if not known_node(vert):
            unknowns.append(vert)
            indices.append(idx)

    # Now for each face in the mesh
    count = 0
    logger.debug("Constructing the global mass matrix")
    for face in obj['faces']:

        count += 1
        logger.debug(" => Processing element #%i" % count)
        logger.debug(" => Element is made of vertices %s" % str(face))
        logger.debug(" => With coordinates: %s" %
                     str([tuple(obj['vertices'][v]) for v in face]))

        # Expand the face to get the coordinates of each node
        co = [(i, j) for i in face for j in face]

        # These are for debug info and matrix construction
        debug_row = []
        debug_col = []
        debug_data = []

        # Add the weights contributed by each point to the global matrix
        for i, j in co:

            # For the mass matrix we only do this for i's that are unknown
            # but for all j's
            if i not in indices:
                continue

            debug_row.append(indices.index(i))
            debug_col.append(j)
            debug_data.append(local_mass[face.index(i)][face.index(j)])

        logger.debug(' => Which gives the follwing entries to the matrix'
                     '\nRows: %s\nCols: %s\nValues: %s'
                     % (debug_row, debug_col, debug_data))

        # Now update the real matrix
        rows = np.append(rows, debug_row)
        cols = np.append(cols, debug_col)
        values = np.append(values, debug_data)

    logger.info('Creating the global mass matrix')
    global_matrix = coo_matrix((values, (rows, cols))).tocsr()

    logger.info(" => %s Global matrix given by: \n%s"
                % (global_matrix.shape, str(global_matrix.toarray())))

    return global_matrix

def construct_global_stiffness_matrix(domain, paramters):
    """
    Given the obj representing the domain and the paramters which
    characterise the system this function constructs the global
    stiffness matrix which we can then use to solve the system.

    Arguments:
        - domain: The parsed obj representing the domain
        - paramters: A tuple containing the paramters a,b,c which
                     characterise the system.
    """

    obj = domain['obj']
    known_node = domain['known_nodes']

    # As a very rough measure of the resolution of the mesh, we will
    # look at two points in a face and determine how large the steps are

    f = obj['faces'][0]
    h = abs((obj['vertices'][f[0]] - obj['vertices'][f[1]])[0])

    logger.info("Setup for the global system:")
    logger.info(" => Domain has %i vertices with %i finite elements"
                % (len(obj['vertices']), len(obj['faces'])))
    logger.info(" => Which implies the value of h to be %.6f" % h)
    logger.info(" => Parameters: a = %.2f, b = [%s, %s], c = %.2f"
                % (paramters[0], paramters[1][0], paramters[1][1],  paramters[2]))

    # Get the local stiffness matrix for this version of the problem
    local_stiffness = local_stiffness_matrix(*paramters, h).A

    logger.info(" => Local Stiffness Matrix given by: \n%s" %
                (str(local_stiffness)))

    # Create numpy arrays for the row and column information
    rows = np.array([], dtype=int)
    cols = np.array([], dtype=int)
    values = np.array([])

    unknowns = []
    indices = []

    for idx, vert in enumerate(obj['vertices']):
        if not known_node(vert):
            unknowns.append(vert)
            indices.append(idx)

    # For each face in the mesh
    count = 0
    logger.debug("Constructing the global stiffness matrix")
    for face in obj['faces']:

        count += 1
        logger.debug(" => Processing element #%i" % count)
        logger.debug(" => Element is made of vertices %s" % str(face))
        logger.debug(" => With coordinates: %s" %
                     str([tuple(obj['vertices'][v]) for v in face]))

        # Expand the face to get the 'co ordinates' each of the nodes
        co = [(i, j) for i in face for j in face]

        # These are for debug info and matrix construction
        debug_row = []
        debug_col = []
        debug_data = []

        # Add the weights to each point i,j in the global matrix
        for i, j in co:

            # Only do this for nodes that are not known
            if i not in indices or j not in indices:
                continue

            debug_row.append(indices.index(i))
            debug_col.append(indices.index(j))
            debug_data.append(
                local_stiffness[face.index(i)][face.index(j)])

        logger.debug(' => Which gives the follwing entries to the matrix'
                     '\nRows: %s\nCols: %s\nValues: %s'
                     % (debug_row, debug_col, debug_data))

        # Now update the actual matrix
        rows = np.append(rows, debug_row)
        cols = np.append(cols, debug_col)
        values = np.append(values, debug_data)

    # Build the coo matrix and convert it to a CSR matrix
    logger.info('Creating the global stiffness matrix')
    global_matrix = coo_matrix((values, (rows, cols))).tocsr()

    logger.info(" => %s Global matrix given by: \n%s"
                % (global_matrix.shape, str(global_matrix.toarray())))

    return global_matrix
