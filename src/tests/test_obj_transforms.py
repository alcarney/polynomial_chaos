import numpy as np

from hypothesis import given
from hypothesis.strategies import floats, tuples, lists, integers

from obj.transforms import cycle_vertices, to_numpy_array, check_face_order

# The 'types' we want hypothesis to generate
coord = floats(min_value=10**-6, max_value=10**6)
index = integers(min_value=1, max_value=10**6)
vertex = tuples(coord, coord, coord)
obj_vertices = lists(elements=vertex, min_size=1, average_size=100)

# @composite
# def right_angled_triangle(draw):
#     """
#     This strategy will randomly generate vertices which define a right
#     angled triangle.
#     """
#     order = draw(choices([(0, 1, 2), (1, 2, 0), (2, 0, 1)]))

#     x = draw(coord)
#     y = draw(coord)
#     dx = draw(coord)
#     dy = draw(coord)

#     v = [np.array([x, y, 0]), np.array([x + dx, y, 0]), np.array([x, y + dy, 0])]

#     return [v[i] for i in order]


# @given(tri=right_angled_triangle())
# def test_check_face_order(tri):

#     f1 = (0, 1, 2)
#     f2 = (1, 2, 0)
#     f3 = (2, 0, 1)

#     assert (0, 1, 2) == check_face_order(f1, tri) == check_face_order(f2, tri) == check_face_order(f3, tri)

# TODO: Get hypothesis to genreate examples for this
def test_check_face_order():
    a = np.array([1.0, 1.0, 0.0])
    b = np.array([4.0, 1.0, 0.0])
    c = np.array([1.0, 2.0, 0.0])
    verts = [a, b, c]

    f1 = (0, 1, 2)
    f2 = (1, 2, 0)
    f3 = (2, 0, 1)

    assert (0, 1, 2) == check_face_order(f1, verts) == check_face_order(f2, verts) == check_face_order(f3, verts)


@given(x=index, y=index, z=index)
def test_cycle_vertices(x, y, z):

    assert (y, z, x) == cycle_vertices((x, y, z))


@given(verts=obj_vertices)
def test_to_numpy_array(verts):

    # Build the obj object as we expect it to be
    obj = {'vertices': verts}

    # Convert each vertex to a numpy array
    new_obj = to_numpy_array(obj)

    # Loop though each list and check the representations are
    # equivalent
    for u, v in zip(new_obj['vertices'], verts):
        assert all(u == v)
