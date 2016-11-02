from hypothesis import given
from hypothesis.strategies import integers, floats

from obj.parser import parse_face, parse_vertex, parse_obj

# Set the type of data that I want hypothesis to generate
coord = floats(min_value=10**-6, max_value=10**6)
index = integers(min_value=1, max_value=10**6)


@given(x=coord, y=coord, z=coord)
def test_parse_coord(x, y, z):

    # We only need a certain amount of precision
    x = round(x, 6)
    y = round(y, 6)
    z = round(z, 6)

    s = "v %.6f %.6f %.6f" % (x, y, z)
    assert parse_vertex(s) == (x, y, z)


@given(f1=index, f2=index, f3=index)
def test_parse_face(f1, f2, f3):

    s = "f %i %i %i" % (f1, f2, f3)

    assert parse_face(s) == (f1 - 1, f2 - 1, f3 - 1)


def test_simple_plane():

    obj = parse_obj('tests/simple_plane.obj')

    assert len(obj['vertices']) == 4
    assert len(obj['faces']) == 2
    assert obj['faces'] == [(1, 2, 0), (1, 3, 2)]
    assert obj['vertices'] == [(-1.0, -1.0, 0.0), (1.0, -1.0, 0.0),
                               (-1.0, 1.0, 0.0), (1.0, 1.0, 0.0)]
