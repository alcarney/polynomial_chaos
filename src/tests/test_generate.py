import numpy as np

from obj.parser import parse_obj
from obj.generate import rectangular_domain
from obj.transforms import to_numpy_array, enforce_fem_order

# When N=4 and x_min = y_min = 0 and x_max = y_max = 0
# we should get an indentical result if we use the
# obj mesh or our generator
def test_rectangular_domain():

    obj = parse_obj('meshes/plane_2d_n4.obj')
    obj = to_numpy_array(obj)
    obj = enforce_fem_order(obj)

    mesh = rectangular_domain()

    vertices = mesh['vertices']
    faces = mesh['faces']

    # First the number of vertices should match
    assert len(vertices) == len(obj['vertices'])

    # Then the order should match
    for i in range(0, len(vertices)):
        assert np.equal(vertices[i], obj['vertices'][i]).all()

    # Now the number of faces should also match
    assert len(faces) == len(obj['faces'])

    # And the order
    for i in range(0, len(faces)):
        assert faces[i] == obj['faces'][i]
