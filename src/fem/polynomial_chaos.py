from itertools import product
from functools import reduce
from scipy.special import legendre


def prod(it):
    """
    Given it, an iterable return the product of all its entries
    """
    return reduce(lambda x, y: x*y, it, 1)


def mk_lengendre_basis(index):
    """
    Given the multi index 'index' construct a function representing
    that particular legendre basis function
    """

    # How many dimensions?
    d = len(index)

    # Construct the function
    def Ps(*args):

        # Add some error checking
        if len(*args) != d:
            raise ValueError('Expected %i arguments!' % d)

        return prod([legendre(n)(x) for n, x in zip(index, *args)])

    # Also add an index field so we can find out which ploynomial it is
    Ps._index = str(index)

    return Ps


def legendre_chaos(d, p):
    """
    Given the dimensionality d and the highest degree of polynomial p
    return the P legendre polynomial basis functions
    """

    # Given p and d, generate all the multi-indicies
    indices = filter(lambda idx: sum(idx) <= p,
                     product(*tuple(range(p+1) for _ in range(d))))

    # Construct the corresponding legendre basis and return
    return [mk_lengendre_basis(idx) for idx in indices]
