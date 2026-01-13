import unittest
import numpy as np
import sys
sys.path.append('../python')
sys.path.append('./python')

from lem import simple_model  
import math
import matplotlib.pyplot as plt
class TestLandscapeEvolution(unittest.TestCase):

    def setUp(self):
        """Set up the test environment."""
        self.model = simple_model()
        self.model.dy = 1.0
        self.model.dx = 1.0
        self.model.m = 0.45
        self.model.k = 1e-10
        self.model.dt = 10
    

    def test_set_z(self):
        """Test the set_z method."""
        new_z = np.random.rand(10, 5)
        ny,nx= np.shape(new_z)
        self.model.set_z(new_z)
        self.assertTrue(np.allclose(self.model.get_z(), new_z))

    def test_set_bc(self):
        """Test the set_bc method."""
        bc_data = np.zeros(np.shape(self.model.get_z()))
        bc_data[:, 0] = 1
        self.assertTrue(bc_data.shape == np.shape(self.model.get_z()))
        self.model.set_bc(bc_data)
        self.assertTrue(np.allclose(self.model.BCX , bc_data))

    def test_slp(self):
        # Test if it's working first with a large DEM
        self.model.slp()
        self.assertTrue(np.all(self.model.slps >=0))
        self.assertTrue(np.mean(self.model.slps)>0)
        # Test accuracy with a small 3x3 example
        self.model.set_z(np.array([[1,0.6,0.3],[1,2,1],[1e-10,2,3]])) #Now test a less random scenario
        self.model.slp()
        self.assertAlmostEqual(self.model.slps[1,1], 2/2**.5) # The slope should be 1.7 / diagonal coords
        self.assertEqual(self.model.receiver[1,1], 2) # The receiver of the center should be upper right

    def test_sinkfill(self):
        # A large array with a pit
        z = np.random.rand(100,100) + 100
        z[40:60,40:60] = 0
        self.model.set_z(z.copy())
        self.model.sinkfill()
        nsinks = self.model.slp() # The slp method returns the number of sinks
        self.assertEqual(nsinks,0)
    def test_stack(self):  
        z_test= np.zeros((5,6),dtype=np.float64)+([ #The hand-calculated stack
                 [0,0,1,0,1,2],
                 [0,2,1,2,1,3],
                 [0,1,3,2,1,4],
                 [0,2,2,1,1.01,5],
                 [0,2,3,3,4,4]])
        
        self.model.set_z(z_test)
        bcx = np.zeros(np.shape(z_test))
        bcx[:, 0] = 1
       
        self.model.set_bc(bcx)
        self.model.slp()
        self.model.stack()
        realstack = np.array([0,5,6,11,12,10,15,
                      16,21,20,25,1,26,2,
                      7,22,17,23,27,3,8,18,13,28,4,9,14,
                      19,24,29])
        self.assertTrue(np.all(self.model.stackij == realstack))

    def test_acc(self):
        """Test the acc method."""
        # This test is difficult to automate fully due to the complex calculations.
        # Instead, we'll check some basic conditions.
        z_test= np.zeros((5,6),dtype=np.float64)+([ #The hand-calculated stack
            [0,0,1,0,1,2],
            [0,2,1,2,1,3],
            [0,1,3,2,1,4],
            [0,2,2,1,1.01,5],
            [0,2,3,3,4,4]])
        
        self.model.set_z(z_test)
        self.model.slp()
        self.model.stack()
        self.model.acc()
        self.assertEqual(self.model.A.max(), 4)

    def test_erode(self):
        """Test the erode method."""
        self.model.dt=.1
        self.model.n = 1.0
        self.model.m = 0.5
        self.model.k = 1e-7

        Z1 = np.random.rand(100,100) *100
        self.model.set_z(Z1.copy())
        self.model.sinkfill()
        Zi = self.model.get_z().copy()
        self.model.slp()
        self.model.stack()
        self.model.acc()
        self.model.erode()

        # Test against the explicit solution - small enough time step should give similar results

        E_fastscape = Zi - self.model.get_z()
        E_explicit = self.model.dt * self.model.k \
            * (self.model.A * self.model.dy * self.model.dx) ** self.model.m \
            * self.model.slps ** self.model.n 



        self.assertTrue(np.allclose(E_fastscape, E_explicit))

        ### Test a different n
        self.model.n = 3
        self.model.m = 1.5
        self.model.dt = .0001
        Z1 = np.random.rand(100,100) *100
        self.model.set_z(Z1.copy())
        self.model.k = 1e-9
        self.model.sinkfill()
        Zi = self.model.get_z().copy()
        self.model.slp()
        self.model.stack()
        self.model.acc()
        self.model.erode()
        E_fastscape = Zi - self.model.get_z()
        E_explicit = self.model.dt * self.model.k \
            * (self.model.A ) ** self.model.m * (self.model.dy * self.model.dx)**self.model.m \
            * self.model.slps ** self.model.n
        print('#n=3.0')

        print(np.mean(E_fastscape))
        print(np.mean(E_explicit))
        self.assertTrue(np.allclose(E_fastscape, E_explicit))

        ### Test a different G
        self.model.n = 1
        self.model.m = 0.45
        self.model.dt = 19
        self.model.G = 5.0
        Z1 = np.random.rand(100, 100) * 100
        self.model.set_z(Z1.copy())
        self.model.k = 1e-7
        self.model.sinkfill()
        Zi = self.model.get_z().copy()
        self.model.slp()
        self.model.stack()
        self.model.acc()
        self.model.erode()
        E_fastscape = Zi - self.model.get_z()
        self.model.set_z(Zi.copy())
        self.model.erode_explicit()
        E_explicit = Zi - self.model.get_z()
        print('#G=1.0')
        print(np.min(E_fastscape[E_fastscape > -999999]))
        print(np.min(E_explicit[E_explicit > -999999]))
        #np.max(E_fastscape)
        #self.assertTrue(np.allclose(E_fastscape, E_explicit))


    def test_erode_explicit(self):
        """Test the erode_explicit method against fs implicit erosion - should be similar for low dt """
        z = np.random.rand(500,500)*1
        BC = np.zeros((500,500))
        BC[:,0]=1
        BC[:,-1]=1
        self.model.set_z(z)
        self.model.set_bc(BC)
        self.model.dt=1000
        self.model.n=1
        self.model.m=0.45
        self.model.k=1e-6


        self.model.sinkfill()
        self.model.slp()
        self.model.stack()
        self.model.acc()
        Zi=self.model.get_z().copy()
        self.model.erode()


        E_implicit=-self.model.get_z() + Zi
        self.model.set_z(Zi.copy())
        self.model.erode_explicit()
        E_explicit = -self.model.get_z() + Zi

        self.assertTrue(np.mean(E_implicit)>1e-17)
        self.assertAlmostEqual(np.mean(E_implicit), np.mean(E_explicit),5)

    def test_diffusion(self):
           
        """Test the diffusion method."""

        ## Analytical solution - square function
        D = 1.0
        self.model.D=D
        x = np.arange(-250,250)
        dt = 1000
        term1 = (x + 50) / math.sqrt(4 * D * dt)
        term2 = (x - 50) / math.sqrt(4 * D * dt)
        z_analytic =  [0.5 * (math.erf(term1[i]) - math.erf(term2[i])) for i in range(len(term1))]

        Zi = np.zeros((500,500))
        Zi[200:300,:]=1
        self.model.set_z(Zi)
        self.model.diffusion()
        z_numeric = self.model.get_z()[:,250]
        self.assertTrue(np.sum(z_analytic-z_numeric)/np.sum(z_analytic)<1e-6) #0.0001% error

if __name__ == '__main__':
    unittest.main()