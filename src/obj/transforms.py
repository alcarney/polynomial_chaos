import numpy as np


def enforce_fem_order(obj):
    """
    This function essentially takes the 'check_face_order' function
    and 'foralls' it over all the faces in the given object
    """
    verts = obj['vertices']

    obj['faces'] = [check_face_order(face, verts) for face in obj['faces']]

    return obj


def check_face_order(face, verts):
    """
    To simplify our code later on, we want the vertices to be listed in
    a particular order, namely counter clockwise starting from the vertex
    on the right angle of the triangle.

    Fortanately for us the .obj exporter from Blender lists the vertices in
    counter clockwise order. Not so fortanately for us the order isn't
    correct for us.

    This function tries to fix that, by taking in a face it will ensure that
    the vertices are indeed in the correct order

    IMPORTANT!! This function assumes that you are representing your vertices
    as numpy arrays

    IMPORTANT!! This function assumes that you have in fact given it a right
    angled triangle. If your face has more than one right angle then it will
    stop after the first one. If no right angle exists then the function will
    not terminate
    """

    # This function will be useful for generating the vectors to test with
    as_vectors = lambda a, b, c: (a - b, a - c)

    # Substitute the actual coordinates in for the
    # vertex index
    vertices = tuple(verts[v] for v in face)

    # Does it start with the vertex as the right angle?
    on_right_angle = False

    while not on_right_angle:

        # check for the right angle
        if np.inner(*as_vectors(*vertices)) != 0.0:
            # => vectors not orthogonal, permute and try again
            vertices = cycle_vertices(vertices)
            face = cycle_vertices(face)  # don't forget the indices as well!
        else:
            # => we found the right angle, return the order
            return face


def cycle_vertices(face):
    """
    Given a face, this permutes the indices in a 'clockwise' direction
    i.e. 1,2,3 -> 2,3,1 -> 3,1,2 -> 1,2,3
    """

    new_order = [1, 2, 0]
    return tuple(face[i] for i in new_order)


def to_numpy_array(obj):
    """
    By default the obj parser returns the vertices as a list of
    tuples, however in some cases it might be useful to consider the
    vertices as numpy arrays. This function will convert each tuple to
    a numpy array
    """

    obj['vertices'] = [np.array(v) for v in obj['vertices']]

    return obj
