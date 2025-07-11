import unittest
import numpy as np
import sys
sys.path.append("./python/")
sys.path.append("../python/")

# Import the priority queue class
from numba_pq import PQ
import sys
import matplotlib.pyplot as plt
import time

class TestPQ(unittest.TestCase):

    def test_init(self):
        # Create an elevation vector
        z = np.float64([5.0,1,2,6,5]) # I include 0 so that the top value is at index 1

        # Initialize the priority queue
        pq = PQ(z)

        # Check that the top value of the queue is correct
        self.assertEqual(pq.top(), -1) # We haven't pushed on anything yet, so the top value should be be -1

    def test_push(self):
        # Create an elevation vector
        z = np.float64([3, 2, 5, 6, 1])

        # Initialize the priority queue
        pq = PQ(z)

        # Push indices of all values of z grid onto queue
        for i in range(len(z)):
            pq.push(i)

        # Check that the number of elements in the queue is correct
        self.assertEqual(len(pq.get()), 5)

        # Check that the top value of the queue is correct
        self.assertEqual(z[pq.top()], 1)

        # Check that the ordered z values are correct

    def test_pop(self):
        # Create an elevation vector

        z = np.random.rand(1000)*10

        # Initialize the priority queue
        pq = PQ(z)
        for i in range(len(z)):
            pq.push(i)

        # Pop the lowest value off the queue until empty
        sorted_z = np.zeros(len(z))
        for i in range(len(z)):
            sorted_z[i] = z[pq.top()]
            pq.pop()


        # Check that the ordered z values are correct
        self.assertAlmostEqual(np.sum(np.abs(sorted_z - np.sort(z))), 0.0)


    def test_push_pop(self):
        """Test a mix of pushing and popping"""

        z = np.random.rand(1000)*1000
        # Initialize the priority queue
        pq = PQ(z)


        # Pop the lowest value off the queue until empty
        sorted_z = np.zeros(100000)
        pushed=0
        for i in range(len(z)):
            pq.push(i)
            pushed+=1
        zmin=np.argmin(z)

        
        for i in range(len(sorted_z)):
            randi = np.random.randint(len(z))

            rand_thres = np.random.rand()

            if (rand_thres > .8) & (pushed>0):
                pq.pop()
                sorted_z[i] = z[zmin]
                pushed-=1
                zmin = pq.top()

            elif(pushed>1):
                upperz_idx = np.where(z>z[zmin])
                if len(upperz_idx[0])>0:
                    randi = np.random.randint(len(upperz_idx[0]))
                    pq.push(upperz_idx[0][randi])
                    pushed+=1
            else:
                break

        # Check that the ordered z values are correct

        sorted_z=sorted_z[sorted_z>0]

        self.assertTrue(np.all(np.diff(sorted_z) >= 0))
        self.assertGreater(np.sum(np.diff(sorted_z) > 0)/len(sorted_z), 0.01)
if __name__ == '__main__':
    unittest.main()