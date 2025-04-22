import numpy as np
from numba.experimental import jitclass
from numba import jit, int64, float64, bool_, int8
import math
from numba_pq import PQ as pq
import warnings
# Define jitclass parameters
### Todo - consistent row-major c style indexing


spec = [
    ('m', float64),
    ('dx', float64),
    ('dy', float64),
    ('t', float64),
    ('dt', float64),
    ('__nx', int64),
    ('__ny', int64),
    ('A', float64[:, :]),
    ('__Z', float64[:, :]),
    ('k', float64),
    ('k_grid', float64[:,:]),
    ('n', float64),
    ('receiver', int64[:, :]),
    ('stackij', int64[:]),
    ('U', float64),
    ('chi', float64[:, :]),
    ('__BCX', int8[:, :]),
    ('__BC', int64[:]),
    ('slps', float64[:, :]),
    ('pour_point', int64[:]),
    ('D',float64),
    ('outlet',float64[:,:]),
    ('__dynamic_bc', bool_)]


@jitclass(spec)
class simple_model:
    def __init__(self):
        # Model parameters
        self.m = 0.45  # Drainage area exponent
        self.n = 1.0
        self.dx = 1000.0  # grid spacing (m)
        self.dy = 1000.0
        self.t = 100e6  # total time (yr)
        self.dt = 1e6  # Time step
        self.__nx = 500  # Number of x grid points
        self.__ny = 500  # Number of y grid points
        self.D = 0.1
        # Data Structures
        self.slps = np.ones((self.__ny, self.__nx), dtype=np.float64)
        self.__Z = np.random.rand(self.__ny, self.__nx) * 10  # Elevation
        self.receiver = np.zeros((self.__ny, self.__nx), dtype=np.int64) #Receiver grid
        self.k =  1e-6  # Erodibility
        self.k_grid = np.zeros((0 , 0))
        self.outlet = np.zeros((self.__ny,self.__nx))
        # Boundary condition grid, 0 = normal 1 = outlet
        self.__BCX = np.zeros(np.shape(self.__Z), dtype=np.int8)
        self.__BCX[:, 0] = 1  # by default set all edges as outlets
        self.__BCX[:, -1] = 1
        self.__BCX[0, :] = 1
        self.__BCX[-1, :] = 1

        self.stackij = np.zeros(0,dtype=np.int64)

        self.__dynamic_bc = False # dynamic baselevel
        # Convert the boundary condition grid to linear (for speed in some
        # cases)
        self.__BC = np.where(self.__BCX == 1)[0]
        self.pour_point = np.int64([-1,-1]) #Pour_point is an (y,x) coordinate of the pour point if applicable - for drainage extraction


    def sinkfill(self):
        """
        Fill pits using the priority flood method of Barnes et al., 2014.
        """
        c = int(0)
        nn = self.__nx * self.__ny
        p = int(0)
        closed = np.full(nn, False)
        pit = np.zeros(nn, dtype=np.int32)
        idx = [1, -1, self.__ny, -self.__ny, -self.__ny + 1, -self.__ny - 1,
               self.__ny + 1, self.__ny - 1]  # Linear indices of neighbors
        open = pq(self.__Z.transpose().flatten())
        for i in range(len(self.__BC)):
            open = open.push(self.__BC[i])
            closed[self.__BC[i]] = True
            c += 1
        for i in range(0, self.__ny):
            for j in range(0, self.__nx):
                if (i == 0) or (j == 0) or (
                        j == self.__nx - 1) or (i == self.__ny - 1):
                    # In this case only edge cells, and those below sea level
                    # (base level) are added
                    ij = j * self.__ny + i
                    if not closed[ij]:
                        closed[ij] = True
                        open = open.push(ij)
                        c += 1
        s = int(0)
        si = int(0)
        ij = int(0)
        ii = int(0)
        jj = int(0)

        pittop = int(-9999)
        count1 = 0
        while ((c > 0) or (p > 0)):
            if ((p > 0) and (c > 0) and (pit[p - 1] == -9999)):
                s = open.top()

                open = open.pop()  # The pq class (above) has seperate methods for pop and top (others may combine both functions)
                c -= 1
                pittop = -9999
            elif p > 0:
                s = int(pit[p - 1])
                pit[p - 1] = -9999
                p -= 1

                if pittop == -9999:
                    si, sj = self.lind(s, self.__ny)
                    pittop = self.__Z[si, sj]
            else:
                s = int(open.top())
                open = open.pop()
                c -= 1
                pittop = -9999
            si, sj = self.lind(s, self.__ny)  # Current
            count1 += 1

            for i in range(8):
                ij = idx[i] + s
                ii, jj = self.lind(ij, self.__ny)  # Neighbor
                if ((ii >= 0) and (jj >= 0) and (
                        ii < self.__ny) and (jj < self.__nx)):
                    if not closed[ij]:
                        closed[ij] = True
                        if self.__Z[ii, jj] <= self.__Z[si, sj]:
                            # This (e) is sufficiently small for most DEMs but
                            # it's not the lowest possible.  In case we are
                            # using 32 bit, I keep it here.
                            self.__Z[ii, jj] = self.__Z[si, sj] + 1e-8 * np.random.rand() + 1e-6 #the e value with some randomness- we can adjust this
                            pit[p] = ij
                            p += 1
                        else:
                            open = open.push(ij)
                            c += 1
        return

    def lind(self, xy, n:float64):
        """
        compute bilinear index from linear indices - trivial but widely used (hence the separate function)

        :param xy:  linear index
        :param n: ny or nx (depending on row-major or col-major indexing)
        :return:
        """
        x = math.floor(xy / n)
        y = xy % n
        return y, x
    
    def turn_on_off_dynamic_bc(self, dynamic_bc):
        self.__dynamic_bc = dynamic_bc
        if self.__dynamic_bc:
            self.__BCX = self.__Z <= 0
            self.__BC = np.where(self.__BCX.transpose().ravel() == 1)[0]
            
        return
    
    def set_bc(self, bc:int8[:,:]):
        """
        Set the boundary conditions

        :param bc: Boundary condition grid 1 = outlet node 0 = non-outlet. 
        Must be same size as Z

        """
        ny, nx = np.shape(bc)
        print([ny,nx])
        print(self.__ny,self.__nx)
        print(np.shape(self.__Z))
        if (ny, nx) != np.shape(self.__Z):
            raise ValueError("Wrong size for Boundary Condition grid."
            " bc matrix must be same size as Z. Maybe you have not yet set Z?")
        if np.any(bc.ravel() > 0):
            self.__BCX = np.zeros(np.shape(self.__Z), dtype=np.int8)
            # Have to do this loop because of unresolved type casting issues with numba and int8
            for i in range(ny):
                for j in range(nx):
                    if bc[i,j] > 0:
                        self.__BCX[i,j] = 1
            self.__BC = np.where(self.__BCX.ravel() == 1)[0]

        return

    def set_z(self, Z:float64[:,:]):
        """
        :param Z: New elevation grid

        Set the elevation and resizes other grids correspondingly
        """
        ny, nx = np.shape(Z)
        try:
            self.__Z = Z
        except:
            raise ValueError("Z is not np.float64")
        if (self.__ny, self.__nx) != (ny, nx):
            if len(self.k_grid)>0:
                self.k_grid = np.zeros((ny, nx), dtype=np.float64) + self.k
                print("k grid resized and reset to default k value")

            self.slps = np.zeros((ny, nx), dtype=np.float64)
            self.__BCX = np.zeros((ny, nx), dtype=np.int8)
            self.__BCX[:, 0] = 1
            self.__BCX[:, -1] = 1
            self.__BCX[0, :] = 1
            self.__BCX[-1, :] = 1
            self.set_bc(self.__BCX)
            print('Boundary condition values have been reset')

        self.__ny = ny
        self.__nx = nx
        
     
    def get_z(self):
        return self.__Z

    def slp(self):  # Calculate slope and steepest descent
        """
        D8 slopes
        """
        eps = 1e-20 # The e value for sinks i.e. dz from grid point to neighboring grid point within flats
        ij = 0
        c = 0
        self.receiver = np.zeros((self.__ny, self.__nx), dtype=np.int64)
        for i in range(0, self.__ny):
            for j in range(0, self.__nx):
                ij = j * self.__ny + i
                mxi = 0
                self.receiver[i, j] = ij

                if (0 < i < self.__ny and j > 0 and j <
                        self.__nx - 1 and i < self.__ny - 1 and not self.__BCX[i,j]):
                    for i1 in range(-1, 2):
                        for j1 in range(-1, 2):
                            mp = (self.__Z[i, j] - self.__Z[i + i1, j + j1]) / (np.sqrt(
                                (float(i1 * self.dy) ** 2) + float(j1 * self.dx) ** 2) + eps)  # In case slope iz zero, we add 1e-10 to ensure no div by 0
                            if mp  > mxi: 
                                ij2 = (j + j1) * self.__ny + i1 + i
                                mxi = mp

                                self.slps[i, j] = (self.__Z[i, j] - self.__Z[i + i1, j + j1]) / np.sqrt(
                                    (float(i1 * self.dy) ** 2) + float(j1 * self.dx) ** 2 + eps) 
                                self.receiver[i, j] = ij2

                    if mxi == 0:
                        self.outlet[i,j]=1
                        c += 1
        return c


    def slp_basin(self):
        """
        This is a version of the D8 network calculation which excludes adding receivers to
        the stack which are at or below 0 elevation - ideal for basins in which we want to
        remove elements of the landscape that are not part of the basin of interest.
        """
        ij = 0
        c = 0
        fnd = np.zeros((self.__ny, self.__nx))
        self.receiver = np.zeros((self.__ny, self.__nx), dtype=np.int64)
        if self.__dynamic_bc: #We must do this at every step to ensure we have the BCs, the computational cost is low...
            self.__BC = np.where(self.__Z.ravel() <= 0)
            self.__BCX.ravel()[self.__BC] = 1
        for i in range(0, self.__ny):
            for j in range(0, self.__nx):
                ij = j * self.__ny + i
                mxi = 0
                self.receiver[i, j] = ij
                if 0 < i < self.__ny and 0 < j < self.__nx - 1 and i < self.__ny - 1 and self.Z[i,j]>=0:
                    for i1 in range(-1, 2):
                        for j1 in range(-1, 2):
                            if self.__Z[i + i1, j + j1] > 0:
                                mp = (self.__Z[i, j] - self.__Z[i + i1, j + j1]) / np.sqrt(
                                    (float(i1 * self.dy) ** 2) + float(j1 * self.dx) ** 2 + 1e-10)
                                if mp > mxi:
                                    ij2 = (j + j1) * self.__ny + i1 + i
                                    mxi = mp
                                    self.slps[i, j] = (self.__Z[i, j] - self.__Z[i + i1, j + j1]) / np.sqrt(
                                        (float(i1 * self.dy) ** 2) + float(j1 * self.dx) ** 2 + 1e-10)
                                    self.receiver[i, j] = ij2
                    if mxi == 0:
                        c += 1
                        fnd[i, j] = 1
        return fnd

    def stack(self):
        """
        takes the input flowdirs from "receiver" grid and makes the topologically ordered
         stack of the stream network . This is a slightly different approach from the
         Fastscape algorithm which uses a recursive function -
         instead this uses a while loop, which is more efficient.

        :return: topologically ordered stack
        """
        self.stackij = np.zeros(self.__ny * self.__nx, dtype=np.int64)
        if self.pour_point[1] > -1:
            for i in range(0, self.__ny):
                for j in range(0, self.__nx):
                    ij = j * self.__ny + i
                    if self.receiver[i,j] == ij:
                        self.receiver[i,j]=-1
            self.receiver[int(self.pour_point[0]), int(self.pour_point[1])] = self.pour_point[1] * self.__ny + self.pour_point[0]

        c = 0
        k = 0
        for i in range(0, self.__ny):
            for j in range(0, self.__nx):

                ij = j * self.__ny + i
                i2 = i
                j2 = j
                if self.receiver[i, j] == ij:
                    self.stackij[c] = ij
                    c += 1
                    while k < c <= self.__ny * self.__nx - 1:
                        for i1 in range(-1, 2):
                            for j1 in range(-1, 2):
                                if 0 < j2 + j1 < self.__nx - 1 and 0 < i2 + i1 < self.__ny - 1:

                                    ij2 = (j2 + j1) * self.__ny + i2 + i1
                                    if ij != ij2 and self.receiver[i2 + i1, j2 + j1] == ij:
                                        self.stackij[c] = ij2
                                        c += 1
                        k = k + 1
                        ij = self.stackij[k]
                        i2, j2 = self.lind(ij, self.__ny)

    def acc(self, init=np.ones([1, 1])):
        """
        Takes the stack and receiver grids and computes drainage area.

        """
        self.A = np.ones((self.__ny, self.__nx), dtype=np.float64)
        self.A[:, :] = init[:, :]
        for ij in range(len(self.stackij) - 1, 0, -1):
            i, j = self.lind(self.stackij[ij], self.__ny)
            i2, j2 = self.lind(self.receiver[i, j], self.__ny)
            if self.stackij[ij] != self.receiver[i, j]:
                self.A[i2, j2] += self.A[i, j]

    def erode(self):
        """
        Erode using fastscape method
        """
        converge_thres = 1e-8

        dA = (self.dx * self.dy) ** self.m
        if self.n == 1:
            ni = 1
        else:
            ni = 5

        k = self.k
        useKGrid = False
        if len(self.k_grid)  > 0:
            useKGrid = True
        for ij in range(0, len(self.stackij)):

            i, j = self.lind(self.stackij[ij], self.__ny)
            i2, j2 = self.lind(self.receiver[i, j], self.__ny)
            if useKGrid: #If we use variable k in a grid....
               k = self.k_grid[i2, j2]
            if (i2 != i) | (j2 != j):
                dx = np.sqrt((float(i2 - i) * self.dy) ** 2 + (
                    float(j2 - j) * self.dx) ** 2)
                f = k * dA * self.A[i, j] ** self.m * self.dt * \
                    (self.__Z[i, j] - self.__Z[i2, j2])**(self.n - 1) / dx ** self.n
                x = 1
                xl = 99999
                ni = 1
                while np.abs(xl - x) / x > converge_thres:
                    xl = x
                    x = x - (x - 1 + f * x ** self.n ) / \
                        (1 + self.n * f * x ** (self.n - 1 ))
                    ni += 1
                
                self.__Z[i, j] = self.__Z[i2, j2] + x * \
                    (self.__Z[i, j] - self.__Z[i2, j2])
                if self.__Z[i,j]<=0:
                    self.__Z[i,j] = 0

    def erode_explicit(self, a_crit:float64=0):
        """
        Erode using explicit method

        :returns: erosion rate grid
        """
        E = np.zeros((self.__ny, self.__nx))
        k = self.k
        useKGrid = False
        if len(self.k_grid > 0):
            useKGrid = True
        for ij in range(0, len(self.stackij)):

            i, j = self.lind(self.stackij[ij], self.__ny)
            i2, j2 = self.lind(self.receiver[i, j], self.__ny)
            if  useKGrid: #If we use variable k in a grid...
                k = self.k_grid[i2, j2]
            if (i2 != i) | (j2 != j):
                if self.A[i, j] > a_crit:
                    f = self.dt * (self.dx * self.dy) ** self.m
                    E[i, j] = k * f * \
                       self.A[i, j] ** self.m * self.slps[i, j]** self.n
        self.__Z -= E
        # self.__Z[:, -1] = 0
        # self.__Z[:, 0] = 0
        # self.__Z[-1, :] = 0
        # self.__Z[0, :] = 0
        return E

    def chicalc(self, U1:float64=1.0, elev_fact=0):
        """
        "params: U1 = normalized uplift rate to be included in chi calculations"
        "params: elev_fact = elevation factor for rivers that do not start at zero elevation -  Giachetta and Willett report this as 1/32.2"
        Calculate chi based on the inputs
        """

        self.chi = np.zeros((self.__ny, self.__nx), dtype=np.float64)
        dA = (self.dx * self.dy) ** self.m
        U = np.ones((self.__ny, self.__nx))
        U[:, :] = U1
        for ij in range(len(self.I)):
            i, j = self.lind(self.I[ij], self.__ny)
            i2, j2 = self.lind(self.s[i, j], self.__ny)
            ds = np.sqrt(((i - i2) * self.dy)**2 + ((j - j2) * self.dx)**2)
            if (self.s[i2, j2] == self.s[i, j]):
                self.chi[i, j] = self.Z[i, j] * elev_fact
            else:
                self.chi[i, j] = U[i, j] / (self.A[i, j] ** self.m * dA) * ds
                self.chi[i, j] += self.chi[i2, j2]
        return

    def diffusion(self):
        """
        Explicit diffusion for hillslopes

        :param D: Diffusivity

        """
        Z = self.__Z
        courant_t = min(np.array([self.dx**2, self.dy**2])) / (4 * self.D)
        ny, nx = np.shape(Z)
        E = np.zeros((ny, nx))
        t_tot = 0

        while t_tot < self.dt:
            for i in range(1, ny - 1):
                for j in range(1, nx - 1):
                    zijp = Z[i, j + 1]
                    zijm = Z[i, j - 1]
                    zimj = Z[i - 1, j]
                    zipj = Z[i + 1, j]
                    E[i, j] = self.D * ((2 * Z[i, j] -
                                    zipj -
                                    zimj) /
                                   (self.dy ** 2) +
                                   (2 * Z[i, j] - zijp - zijm)
                                   / (self.dx ** 2))

            if self.dt - t_tot < courant_t:
                courant_t = self.dt - t_tot
            E *= courant_t
            t_tot += courant_t
            self.__Z -= E
        return E


# @jit(nopython=True)
# def lind(xy, n):
#     """
#     Non - object oriented version of function for parallelization (Python does not allow pickled JIT class
#     Compute bilinear index from linear indices - trivial but widely used (hence the separate function)

#     :param xy:  linear index
#     :param n: ny or nx (depending on row-major or col-major indexing)
#     :return:
#     """
#     x = math.floor(xy / n)
#     y = xy % n
#     return y, x


# @jit(nopython=True)
# def erode_explicit(
#         slps,
#         I,
#         s,
#         A,
#         E,
#         dx=90,
#         dy=90,
#         m=0.45,
#         n=1.0,
#         k=1e-8,
#         dt=1,
#         carea=0,
#         G=0):
#     """

#     :param G: Transport capacity coefficient of Yuan et al. (2019)
#     :param ny: y grid size
#     :param nx: x grid size
#     :param I: fastscape stack
#     :param s: list of receivers for the stack
#     :dx: x resolution
#     :dy: y resolution
#     :m: Stream power m
#     :n: stream power n
#     :k: stream power k
#     :slps: Grid of slopes for steepest descent
#     :dt: time resolution
#     :A: Grid of drainage areas
#     :E: Erosion rate grid (can be input based on previous result, otherwise set to zero)
#     :carea: critical area
#     :return: Fluvial Erosion map

#     Fluvial erosion using explicit form of transport limited eqn.  Seperated from the main class so that it can
#     be parallelized
#     """
#     ny, nx = np.shape(slps)
#     sedacc = np.zeros((ny, nx))
#     f = (dx * dy) ** m

#     for ij in range(len(I) - 1, 0, -1):

#         i, j = lind(I[ij], ny)
#         i2, j2 = lind(s[i, j], ny)
#         if A[i, j] > carea:
#             E[i, j] += k[i2, j2] * f * \
#                 np.power(A[i, j], m) * np.power(slps[i, j], n) - G * sedacc[i, j] / A[i, j]
#         sedacc[i2, j2] += E[i, j]
#     E *= dt

#     return E

# def smooth(windowSize,I,s,z,acc,athres=5):
    
#     zsd = np.zeros((windowSize,len(I)))
#     zsu = np.zeros((windowSize,len(I)))
#     avgs=np.zeros(len(I))
#     distsU=np.zeros(len(I))
#     distsD = np.zeros(len(I))
#     ns = np.zeros(len(I))
#     amaxs = np.zeros(len(I))
#     for i in range(len(I)):
#         if s[i] != I[i]:
#             zsd[:-1,I[i]] = zsd[1:,s[i]]
#             zsd[-1,I[i]] = z[s[i]]
#     for i in range(len(I)-1,0,-1):
#         if s[i] != I[i]:
#             if (acc[I[i]] >= windowSize) and (acc[I[i]] >= amaxs[s[i]]):
#                 amaxs[s[i]] = acc[I[i]]
#                 zsu[:,s[i]] = zsu[:,I[i]]
#                 zsu[:-1,s[i]] = zsu[1:,I[i]]
#                 zsu[-1,s[i]] = z[I[i]]
#     for i in range(len(I)):
#         lu = len(np.where(zsu[:,i]>0)[0])
#         ld = len(np.where(zsd[:,i]>0)[0])
#         minl = min([lu,ld])
#         if minl>=windowSize:
#             avgs[i] = np.mean(np.concatenate([zsd[:,i][zsd[:,i]>0], zsu[:,i][zsu[:,i]>0]]))
#         elif minl>=1:
#             avgs[i] = np.mean(np.concatenate([zsd[:,i][zsd[:,i]>0][-minl:], zsu[:,i][zsu[:,i]>0][-minl:]]))
#         else:
#             avgs[i] = z[i]
#     return avgs



# @jit(nopython=True)
# def diffuse(Z, D=1.0, dy=90, dx=90, dt=1):
#     """
#     Explicit diffusion for hillslopes

#     :param D: Diffusivity
#     :param Z: Elevation
#     :param dy: x resolution
#     :param dt: time resolution

#     """
#     ny, nx = np.shape(Z)
#     E = np.zeros((ny, nx))
#     for i in range(1, ny - 1):
#         for j in range(1, nx - 1):
#             zijp = Z[i, j + 1]
#             zijm = Z[i, j - 1]
#             zimj = Z[i - 1, j]
#             zipj = Z[i + 1, j]

#             if zijp <= 0:
#                 zijp = Z[i, j]
#             if zijm <= 0:
#                 zijm = Z[i, j]
#             if zimj <= 0:
#                 zimj = Z[i, j]
#             if zipj <= 0:
#                 zipj = Z[i, j]

#             E[i, j] = D * ((2 * Z[i, j] -
#                             zipj -
#                             zimj) /
#                            (dy ** 2) +
#                            (2 * Z[i, j] - zijp - zijm)
#                            / (dx ** 2))

#     E *= dt

#     return E


# @jit(nopython=True)
# def acc(I, s, init=1):
#     """
#     Calculate drainage area or sum some input quantity (e.g. sediment) along the stack

#     :param init: Initial quantity to sum (default is ones)

#     """
#     ny, nx = np.shape(s)
#     A = np.ones((ny, nx))

#     if len(init) >= 1:
#         A[:, :] = init[:, :]
#     for ij in range(len(I) - 1, 0, -1):
#         i, j = lind(I[ij], ny)
#         i2, j2 = lind(s[i, j], ny)
#         if I[ij] != s[i, j]:
#             A[i2, j2] += A[i, j]
#     return A


# if __name__ == '__main__':
#     # An example run
#     F = simple_model()
#     fig = plt.figure()

#     for t in range(0, int(F.t / F.dt)):  # main loop
#         start = timeit.default_timer()
#         F.sinkfill()
#         F.slp()
#         F.stack()
#         F.acc()
#         F.erode()
#         F.Z += 1
#         F.Z[:, 0] = 0
#         F.Z[:, -1] = 0
#         F.Z[0, :] = 0
#         F.Z[-1, :] = 0
#         end = timeit.default_timer()

#         a = plt.imshow(F.Z)
#         plt.colorbar(a)

#         plt.pause(.05)
#         plt.clf()

