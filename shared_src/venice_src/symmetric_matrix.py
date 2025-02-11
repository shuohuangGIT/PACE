import numpy as np


class SymmetricMatrix (object):
    '''
    Symmetric matrix with AMUSE units
    At the moment only meant for reading/writing, algebra is NOT possible!
    '''

    def __init__ (self, N, unit):

        self._len = N
        self._storage = np.zeros( N+(N*(N-1))//2 )  # Data saved as 1D list
        self._unit = unit


    def _mapindex (self, i, j):
        '''
        Map matrix indices to position in reduced list
        Imagine diagonal + upper triangle of matrix; elements are ordered
        left-to-right, top-to-bottom
        '''
        if i >= self._len or j >= self._len:
            raise ValueError("Index exceeded matrix size!")

        # Pick min/max index, this ensures symmetry
        if j >= i:
            P = i
            Q = j
        else:
            P = j
            Q = i

        # Mapping of min/max to position in 1D list
        return P*self._len - (P*(P-1))//2 + Q-P


    def __setitem__ (self, ind, data):
        self._storage[self._mapindex(ind[0],ind[1])] = data.value_in(self._unit)


    def __getitem__ (self, ind):
        return self._storage[self._mapindex(ind[0],ind[1])] | self._unit


    def __len__ (self):
        return self._len*self._len


if __name__ == '__main__':

    from amuse.units import units

    A = SymmetricMatrix(5, units.s)
    A[1,0] = 2. | units.s
    print (A[0,1].value_in(units.s))
