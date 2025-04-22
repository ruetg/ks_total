from numba import int64, float64
from numba.experimental import jitclass
import numpy as np
# Define the specification of the priority queue class
spec2 = [
    ("__numel", int64),
    ("__u", int64),
    ("__ul", int64),
    ("__indices", int64[:]),
    ("__z", float64[:]),
]

# Define the priority queue class using jitclass
@jitclass(spec2)
class PQ:
    """
    A class for jit enabled priority queue.  This serves a fairly specific role at the moment in that
    it allows for an elevation vector (z) to be added.  However it remains unsorted until all of its indices are pushed onto the queue.
    Additionally, the returned values are only the indices.
    For example if z = [3,2,3], add each element sequetially to the vector for i in range(len(z)): pq = pq.push(i) will sort to
    2,3,3.  But then pa.top() will return (1) instead of (2) in order to be used in the Barnes (2014) algorithm.
    In the future it should be more generalized.

    """

    def __init__(self, z):
        """
        initiate all values to 0

        :nn: The indices of z
        :numel: number of elements currently in the queue
        :u:

        """

        self.__numel = np.int64(0)
        self.__u = np.int64(0)  # End of the right side of the queue
        self.__ul = np.int64(0)  # End of the left side of the queue
        self.__indices = np.full(len(z) + 1, 0) # This is the main vector containing indices to be sorted
        self.__z = np.concatenate((np.zeros(1).ravel(), z.ravel()))  # contains the values to be sorted up-descending

    def top(self):
        """
        Get the top value of the queue (lowest value)

        :return:  self
        """
        return self.__indices[1] - 1

    def get(self):
        """
        Get the ordered z values, not necessarily perfectly sorted due to the nature of pq

        :return: ordered z values (lowest to highest)
        """
        zt =  self.__z[self.__indices]
        return zt[zt>0]

    # @property
    def pop(self):
        """
        Pop lowest value off the queue and re-sort

        :return: self
        """
        self.__indices[1] = self.__indices[self.__numel]  # Move the last value to the top and re-sort
        self.__indices[self.__numel] = 0
        self.__u = 2  # End of right hand side (initially we just have 2 sides with 1 element each)
        self.__ul = np.int64(self.__u / 2)  # end of left hand side
        if self.__numel==3:
            if self.__z[self.__indices[2]] < self.__z[self.__indices[1]]:
                t = self.__indices[2]
                self.__indices[2] = self.__indices[1]
                self.__indices[1] = t
            self.__numel -= 1
            return self

        
        while self.__u <= self.__numel-1:
            # Is the end of the current right side less than the end of the next left side? If so, we stay with the current set of sides
            m = self.__u
            if  (self.__u < self.__numel-1) & (self.__z[self.__indices[self.__u]] > self.__z[self.__indices[self.__u + 1]]):
                m = self.__u+1
            if self.__z[self.__indices[self.__ul]] > self.__z[self.__indices[m]]:
                
                t = self.__indices[m]

                self.__indices[m] = self.__indices[self.__ul]
                self.__indices[self.__ul]=t
                self.__u = 2 * (m)
                self.__ul = np.int64(self.__u / 2)

            else:

                break


        self.__numel -= 1


        return self

    def push(self, i):
        """
        Push a value onto the queue (and sort)

        :param i: value to add
        :return: self
        """
        #if (i>len(self.__z)-1) or (i<0):
            #raise ValueError("Value {} for index is out of bounds of the z vector provided".format(str(i)))
        i += 1
        self.__numel += 1
        if self.__numel>len(self.__indices)-1:
            # We ran out of room, so we extend the length in chunks.
            # Must do it this clunky way in numba...
            temparray =  np.zeros(len(self.__indices)*2+1,dtype=np.int64)
            temparray[:len(self.__indices)] = self.__indices
            self.__indices = temparray
        
        self.__u = self.__numel  # The end of the right side of the queue
        self.__ul = np.int64(self.__u / 2)  # The end of the left side of the queue

        self.__indices[self.__u] = i  # initially add index to the end of the right-hand side

        while self.__ul > 0:
            # If end left is greater than end right, switch end left and end right.
            if self.__z[self.__indices[self.__ul]] >= self.__z[self.__indices[self.__u]]:

                t = self.__indices[self.__ul]
                self.__indices[self.__ul] = self.__indices[self.__u]
                self.__indices[self.__u] = t

            else:
                break
            # Now break up the current left hand side into new halves, repeat).
            self.__u = np.int64(self.__u / 2)
            self.__ul = np.int64(self.__u / 2)

        return self