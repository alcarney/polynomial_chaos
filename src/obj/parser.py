

def parse_vertex(vertex_string):
    """
    Parses a vertex definition from an .obj file and returns the tuple
    containing the coordinate of the vertex.
    """

    # Split the string on the spaces
    splits = vertex_string.split(' ')

    # Convert the vertices to floats and store them in a tuple
    return tuple(map(lambda v: float(v), splits[1:]))


def parse_face(face_string):
    """
    Parses a face definition from an .obj file and returns the tuple
    containing the indices of vertcies which comprise that face

    Now in an .obj file the vertices are indexed from 1, whereas python
    indexes things from zero so we will have to shift the numbers by 1 to
    componsate.
    """

    # Split the string on the spaces
    splits = face_string.split(' ')

    # Cast the indices to floats and store them in a tuple
    return tuple(map(lambda f: int(f) - 1, splits[1:]))


def parse_obj(filename):
    """
    Given a filename this function will load in the file and parse
    the .obj file into a data structure we can work with. Namely a
    dictionary containing the following fields:

    - vertices: A list of 3-tuples representing the position of the vertex
                in R^3

    - faces:    A list of 3-tuples representing the indices of the vertices
                which make up the face.
    """

    # Read in the file and split it by line
    with open(filename) as f:

        lines = f.read().split('\n')[:-1]

    # Collect all the lines that start with a 'v ' - these will be the vertices
    # make sure we don't accidentally include the texture coordintes! (vt)
    verts = filter(lambda s: s[0:2] == 'v ', lines)

    # Also collect all the lines that start with a 'f' - these will be the
    # faces
    faces = filter(lambda s: s[0] == 'f', lines)

    # Parse the definitions
    vertices = [parse_vertex(line) for line in verts]
    faces = [parse_face(line) for line in faces]

    # Return the object
    return {'vertices': vertices, 'faces': faces}
