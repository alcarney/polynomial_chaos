import numpy as np

from hypothesis import given
from hypothesis.strategies import floats

from fem.definitions import local_stiffness_matrix


# Some 'types' we want hypothesis to generate
parameter = floats(min_value=10**-6, max_value=10**6)


@given(h=parameter)
def test_local_stiffness_matrix_a_independent_of_h(h):
    """
    With only a non zero the local stiffness matrix should be
    independant of the paramter h
    """

    m = local_stiffness_matrix(1, np.array([0, 0]), 0, h)

    assert np.array_equal(m, np.matrix([[1, -1/2, -1/2],
                                        [-1/2, 1/2, 0],
                                        [-1/2, 0, 1/2]]))
