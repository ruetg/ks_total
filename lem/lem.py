import numpy as np
from numba import jit
from numba.experimental import jitclass
from numba import int64, float64
import matplotlib.pyplot as plt
import math
import timeit

# Define jitclass parameters
spec2 = [
    ('__nn', int64),
    ('__numel', int64),
    ('__u', int64),
    ('__uu', int64),
    ('__ul', int64),
    ('__indices', int64[:]),
    ('__z', float64[:])
]


@jitclass(spec2)
class pq:
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
        self.__nn = np.int64(len(z))
        self.__numel = np.int64(0)
        self.__u = np.int64(0)  # End of the right side of the queue
        self.__uu = np.int64(0)  # Top value of the queue
        self.__ul = np.int64(0)  # End of the left side of the queue
        # This is the main vector containing indices to be sorted
        self.__indices = np.full(len(z) + 1, 0)
        # contains the values to be sorted upward
        self.__z = np.concatenate((np.zeros(1).ravel(), z.ravel()))

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
        return self.__z[self.__indices]

    # @property
    def pop(self):
        """
        Pop lowest value off the queue and re-sort

        :return: self
        """
        self.__uu = self.__indices[1]  # Absolute top value of the queue
        # Move the last value to the top and re-sort
        self.__indices[1] = self.__indices[self.__numel]
        self.__indices[self.__numel] = 0
        # End of right hand side (initially we just have 2 sides with 1 element
        # each)
        self.__u = 2
        self.__ul = np.int64(self.__u / 2)  # end of left hand side
        if self.__numel == 3:

            if self.__z[self.__indices[2]] < self.__z[self.__indices[1]]:
                t = self.__indices[2]
                self.__indices[2] = self.__indices[1]
                self.__indices[1] = t
            self.__numel -= 1

            return self

        while self.__u <= self.__numel - 1:
            # Is the end of the current right side less than the end of the
            # next left side? If so, we stay with the current set of sides
            m = self.__u
            if (self.__u < self.__numel - 1) & (
                    self.__z[self.__indices[self.__u]] > self.__z[self.__indices[self.__u + 1]]):
                m = self.__u + 1
            if self.__z[self.__indices[self.__ul]
                        ] > self.__z[self.__indices[m]]:

                t = self.__indices[m]

                self.__indices[m] = self.__indices[self.__ul]
                self.__indices[self.__ul] = t
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

        i += 1
        self.__numel += 1

        self.__u = self.__numel  # The end of the right side of the queue
        # The end of the left side of the queue
        self.__ul = np.int64(self.__u / 2)

        # initially add index to the end of the right-hand side
        self.__indices[self.__u] = i

        while self.__ul > 0:
            # If end left is greater than end right, switch end left and end
            # right.
            if self.__z[self.__indices[self.__ul]
                        ] >= self.__z[self.__indices[self.__u]]:

                t = self.__indices[self.__ul]
                self.__indices[self.__ul] = self.__indices[self.__u]
                self.__indices[self.__u] = t

            else:
                break
            # Now break up the current left hand side into new halves, repeat).
            self.__u = np.int64(self.__u / 2)
            self.__ul = np.int64(self.__u / 2)

        return self


#
spec = [
    ('m', float64),
    ('dx', float64),
    ('dy', float64),
    ('t', float64),
    ('dt', float64),
    ('nx', int64),
    ('ny', int64),
    ('A', float64[:, :]),
    ('Z', float64[:, :]),
    ('k', float64[:, :]),
    ('n', float64),
    ('s', int64[:, :]),
    ('I', int64[:]),
    ('U', float64),
    ('chi', float64[:, :]),
    ('BCX', int64[:, :]),
    ('BC', int64[:]),
    ('slps', float64[:, :]),
    ('pour_point', int64[:]),
    ('dynamic_bc', float64)]


@jitclass(spec)
class simple_model:
    def __init__(self):
        # Modify default parameters here
        self.m = .5  # Drainage area exponent
        self.n = 1
        self.dx = 1000.0  # grid spacing (m)
        self.dy = 1000.0
        self.t = 100e6  # total time (yr)
        self.dt = 1e6  # Time step
        self.nx = 500  # Number of x grid points
        self.ny = 500  # Number of y grid points
        self.slps = np.ones((self.ny, self.nx), dtype=np.float64)
        self.Z = np.random.rand(self.ny, self.nx) * 10  # Elevation
        self.s = np.zeros((self.ny, self.nx), dtype=np.int64)
        self.k = np.ones(
            (self.ny, self.nx), dtype=np.float64) * 1e-6  # Erodibility
        self.U = .000  # Uplift rate
        # Boundary condition grid, 0 = normal 1 = outlet
        self.BCX = np.full(np.shape(self.Z), 0)
        self.BCX[:, 0] = 1  # by default set all edges as outlets
        self.BCX[:, -1] = 1
        self.BCX[0, :] = 1
        self.BCX[-1, :] = 1
        self.dynamic_bc = -9999 # dynamic baselevel
        # Convert the boundary condition grid to linear (for speed in some
        # cases)
        self.BC = np.where(self.BCX == 1)[0]
        self.pour_point = np.int64([-1,-1]) #Pour_point is an (y,x) coordinate of the pour point if applicable

    def sinkfill(self):
        """
        Fill pits using the priority flood method of Barnes et al., 2014.
        """
        if self.dynamic_bc != -9999:
            bc = np.where(self.Z.ravel() <= 0)
            self.BCX.ravel()[bc] = 1
        c = int(0)
        nn = self.nx * self.ny
        p = int(0)
        closed = np.full(nn, False)
        pit = np.zeros(nn, dtype=np.int32)
        idx = [1, -1, self.ny, -self.ny, -self.ny + 1, -self.ny - 1,
               self.ny + 1, self.ny - 1]  # Linear indices of neighbors
        open = pq(self.Z.transpose().flatten())
        for i in range(len(self.BC)):
            open = open.push(self.BC[i])
            closed[self.BC[i]] = True
            c += 1
        for i in range(0, self.ny):
            for j in range(0, self.nx):
                if (i == 0) or (j == 0) or (
                        j == self.nx - 1) or (i == self.ny - 1):
                    # In this case only edge cells, and those below sea level
                    # (base level) are added
                    ij = j * self.ny + i
                    if not closed[ij]:
                        closed[ij] = True
                        open = open.push(ij)
                        c += 1
        s = int(0)
        si = int(0)
        ij = int(0)
        ii = int(0)
        jj = int(0)
        ci = int(0)
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
                    si, sj = self.lind(s, self.ny)
                    pittop = self.Z[si, sj]
            else:
                s = int(open.top())
                open = open.pop()
                c -= 1
                pittop = -9999
            si, sj = self.lind(s, self.ny)  # Current
            count1 += 1

            for i in range(8):

                ij = idx[i] + s

                ii, jj = self.lind(ij, self.ny)  # Neighbor

                if ((ii >= 0) and (jj >= 0) and (
                        ii < self.ny) and (jj < self.nx)):

                    if not closed[ij]:
                        closed[ij] = True

                        if self.Z[ii, jj] <= self.Z[si, sj]:
                            # This (e) is sufficiently small for most DEMs but
                            # it's not the lowest possible.  In case we are
                            # using 32 bit, I keep it here.
                            self.Z[ii, jj] = self.Z[si, sj] + \
                                1e-8 * np.random.rand() + 1e-6

                            pit[p] = ij
                            p += 1
                        else:
                            open = open.push(ij)
                            c += 1
        return

    @staticmethod
    def lind(xy, n):
        """
        compute bilinear index from linear indices - trivial but widely used (hence the separate function)

        :param xy:  linear index
        :param n: ny or nx (depending on row-major or col-major indexing)
        :return:
        """
        x = math.floor(xy / n)
        y = xy % n
        return y, x

    def set_z(self, Z):
        """
        :param Z: New elevation grid

        Set the elevation and resizes other grids correspondingly
        """
        self.ny, self.nx = np.shape(Z)
        self.Z = Z
        self.k = np.zeros((self.ny, self.nx),
                          dtype=np.float64) + self.k[0, 0]
        self.slps = np.zeros((self.ny, self.nx), dtype=np.float64)
        self.BCX = np.zeros((self.ny, self.nx), dtype=np.int64)
        self.BCX[:, 0] = 1
        self.BCX[:, -1] = 1
        self.BCX[0, :] = 1
        self.BCX[-1, :] = 1
        self.BC = np.where(self.BCX == 1)[0]

    def slp(self):  # Calculate slope and steepest descent
        """
        D8 slopes
        """
        ij = 0
        c = 0
        self.s = np.zeros((self.ny, self.nx), dtype=np.int64)
        if self.dynamic_bc != -9999 :
            print('here')
            bc = np.where(self.Z.ravel() <= 0)
            self.BCX.ravel()[bc] = 1
        for i in range(0, self.ny):
            for j in range(0, self.nx):
                ij = j * self.ny + i
                mxi = 0

                self.s[i, j] = ij
                if (0 < i < self.ny and j > 0 and j < self.nx -
                        1 and i < self.ny - 1 and not self.BCX[i, j]):
                    for i1 in range(-1, 2):
                        for j1 in range(-1, 2):
                            mp = (self.Z[i, j] - self.Z[i + i1, j + j1]) / np.sqrt(
                                (float(i1 * self.dy) ** 2) + float(j1 * self.dx) ** 2 + 1e-10)
                            if mp + 1e-30 > mxi:
                                ij2 = (j + j1) * self.ny + i1 + i
                                mxi = mp

                                self.slps[i, j] = (self.Z[i, j] - self.Z[i + i1, j + j1]) / np.sqrt(
                                    (float(i1 * self.dy) ** 2) + float(j1 * self.dx) ** 2 + 1e-10)
                                self.s[i, j] = ij2

                    if mxi == 0:
                        c += 1

    def set_bc(self, bc):
        """
        Set the boundary conditions

        :param bc: Boundary conditions 1 = outlet node 0 = non-outlet

        """

        self.BC = np.where(bc.ravel() == 1)[0]
        self.BCX = bc

    def slp_basin(self):
        """
        This is a version of the D8 network calculation which excludes adding receivers to
        the stack which are at or below 0 elevation - ideal for basins in which we want to
        remove elements of the landscape that are not part of the basin of interest.
        """
        ij = 0
        c = 0
        fnd = np.zeros((self.ny, self.nx))
        self.s = np.zeros((self.ny, self.nx), dtype=np.int64)
        self.slps = np.zeros((self.ny, self.nx), dtype=np.float64)
        if self.dynamic_bc != -9999 :
            print('here')
            bc = np.where(self.Z.ravel() <= 0)
            self.BCX.ravel()[bc] = 1
        for i in range(0, self.ny):
            for j in range(0, self.nx):
                ij = j * self.ny + i
                mxi = 0
                self.s[i, j] = ij
                if 0 < i < self.ny and 0 < j < self.nx - 1 and i < self.ny - 1:
                    for i1 in range(-1, 2):
                        for j1 in range(-1, 2):
                            if self.Z[i + i1, j + j1] > 0:
                                mp = (self.Z[i, j] - self.Z[i + i1, j + j1]) / np.sqrt(
                                    (float(i1 * self.dy) ** 2) + float(j1 * self.dx) ** 2 + 1e-10)
                                if mp > mxi:
                                    ij2 = (j + j1) * self.ny + i1 + i
                                    mxi = mp
                                    self.slps[i, j] = (self.Z[i, j] - self.Z[i + i1, j + j1]) / np.sqrt(
                                        (float(i1 * self.dy) ** 2) + float(j1 * self.dx) ** 2 + 1e-10)
                                    self.s[i, j] = ij2
                    if mxi == 0:
                        c += 1
                        fnd[i, j] = 1
        print(c)
        return fnd

    def stack(self):
        """
        takes the input flowdirs s and makes the topologically ordered
         stack of the stream network .  This is a slightly different approach from the
         Fastscape algorithm which uses a recursive function - instead this uses a while loop, which is more efficient.
        

        :return: topologically ordered stack
        """
        self.I = np.zeros(self.ny * self.nx, dtype=np.int64)
        if self.pour_point[1] > -1:
            for i in range(0, self.ny):
                for j in range(0, self.nx):
                    ij = j * self.ny + i
                    if self.s[i,j] == ij:
                        self.s[i,j]=-1
            self.s[int(self.pour_point[0]), int(self.pour_point[1])] = self.pour_point[1] * self.ny + self.pour_point[0]

        c = 0
        k = 0
        for i in range(0, self.ny):
            for j in range(0, self.nx):

                ij = j * self.ny + i
                i2 = i
                j2 = j
                if self.s[i, j] == ij:

                    self.I[c] = ij
                    c += 1

                    while k < c <= self.ny * self.nx - 1:

                        for i1 in range(-1, 2):
                            for j1 in range(-1, 2):
                                if 0 < j2 + j1 < self.nx - 1 and 0 < i2 + i1 < self.ny - 1:

                                    ij2 = (j2 + j1) * self.ny + i2 + i1

                                    if ij != ij2 and self.s[i2 +
                                                            i1, j2 + j1] == ij:
                                        self.I[c] = ij2
                                        c += 1

                        k = k + 1
                        ij = self.I[k]
                        i2, j2 = self.lind(ij, self.ny)

    def acc(self, init=np.ones([1, 1])):
        """
        Takes the stack and receiver grids and computes drainage area.

        """
        self.A = np.ones((self.ny, self.nx), dtype=np.float64)
        self.A[:, :] = init[:, :]
        for ij in range(len(self.I) - 1, 0, -1):
            i, j = self.lind(self.I[ij], self.ny)
            i2, j2 = self.lind(self.s[i, j], self.ny)
            if self.I[ij] != self.s[i, j]:
                self.A[i2, j2] += self.A[i, j]

    def erode(self):
        """
        Erode using fastscape method
        """

        dA = (self.dx * self.dy) ** self.m
        if self.n == 1:
            ni = 1
        else:
            ni = 5
        for ij in range(0, len(self.I)):

            i, j = self.lind(self.I[ij], self.ny)
            i2, j2 = self.lind(self.s[i, j], self.ny)
            if (i2 != i) | (j2 != j):
                dx = np.sqrt((float(i2 - i) * self.dy) ** 2 + (
                    float(j2 - j) * self.dx) ** 2)
                f = self.k[i2, j2] * dA * self.A[i, j] ** self.m * self.dt * \
                    (self.Z[i, j] - self.Z[i2, j2])**(self.n - 1) / dx**self.n
                x = 1
                xl = 99999
                ni = 1
                while np.abs(xl - x) / x > 1e-4:
                    xl = x
                    x = x - (x - 1 + f * x**self.n) / \
                        (1 + self.n * f * x**(self.n - 1))

                    ni += 1
                self.Z[i, j] = self.Z[i2, j2] + x * \
                    (self.Z[i, j] - self.Z[i2, j2])

    def erode_explicit(self, a_crit=0):  
        """
        Erode using explicit method

        :returns: erosion rate grid
        """
        E = np.zeros((self.ny, self.nx))
        for ij in range(0, len(self.I)):

            i, j = self.lind(self.I[ij], self.ny)
            i2, j2 = self.lind(self.s[i, j], self.ny)

            if self.A[i, j] > a_crit:
                f = self.dt * (self.dx * self.dy) ** self.m
                E[i, j] = self.k[i2, j2] * f * \
                    np.power(self.A[i, j], self.m) * np.power(self.slps[i, j], self.n)
        self.Z[:, -1] = 0
        self.Z[:, 0] = 0
        self.Z[-1, :] = 0
        self.Z[0, :] = 0
        return E

    def chicalc(self, U1=1.0, elev_fact=0):  
        """
        "params: U1 = normalized uplift rate to be included in chi calculations"
        "params: elev_fact = elevation factor for rivers that do not start at zero elevation -  Giachetta and Willett report this as 1/32.2"
        Calculate chi based on the inputs
        """

        self.chi = np.zeros((self.ny, self.nx), dtype=np.float64)
        dA = (self.dx * self.dy) ** self.m
        U = np.ones((self.ny, self.nx))
        U[:, :] = U1
        for ij in range(len(self.I)):
            i, j = self.lind(self.I[ij], self.ny)
            i2, j2 = self.lind(self.s[i, j], self.ny)
            ds = np.sqrt(((i - i2) * self.dy)**2 + ((j - j2) * self.dx)**2)
            if (self.s[i2, j2] == self.s[i, j]):
                self.chi[i, j] = self.Z[i, j] * elev_fact
            else:
                self.chi[i, j] = U[i, j] / (self.A[i, j] ** self.m * dA) * ds
                self.chi[i, j] += self.chi[i2, j2]
        return

    def diffusion(self, D=.000):
        """
        Explicit diffusion for hillslopes

        :param D: Diffusivity
        :param Z: Elevation
        :param dy: y resolution
        :param dt: time resolution

        """
        Z = self.Z
        courant_t = min(np.array([self.dx**2, self.dy**2])) / (4 * D)
        ny, nx = np.shape(Z)
        E = np.zeros((ny, nx))
        t_tot = 0

        while t_tot < self.dt:
            print(courant_t)
            for i in range(1, ny - 1):
                for j in range(1, nx - 1):
                    zijp = Z[i, j + 1]
                    zijm = Z[i, j - 1]
                    zimj = Z[i - 1, j]
                    zipj = Z[i + 1, j]
                    E[i, j] = D * ((2 * Z[i, j] -
                                    zipj -
                                    zimj) /
                                   (self.dy ** 2) +
                                   (2 * Z[i, j] - zijp - zijm)
                                   / (self.dx ** 2))

            if self.dt - t_tot < courant_t:
                courant_t = self.dt - t_tot
            E *= courant_t
            t_tot += courant_t
            self.Z -= E
        return E


@jit(nopython=True)
def lind(xy, n):
    """
    Non - object oriented version of function for parallelization (Python does not allow pickled JIT class
    Compute bilinear index from linear indices - trivial but widely used (hence the separate function)

    :param xy:  linear index
    :param n: ny or nx (depending on row-major or col-major indexing)
    :return:
    """
    x = math.floor(xy / n)
    y = xy % n
    return y, x


@jit(nopython=True)
def erode_explicit(
        slps,
        I,
        s,
        A,
        E,
        dx=90,
        dy=90,
        m=0.45,
        n=1.0,
        k=1e-8,
        dt=1,
        carea=0,
        G=0):
    """

    :param G: Transport capacity coefficient of Yuan et al. (2019)
    :param ny: y grid size
    :param nx: x grid size
    :param I: fastscape stack
    :param s: list of receivers for the stack
    :dx: x resolution
    :dy: y resolution
    :m: Stream power m
    :n: stream power n
    :k: stream power k
    :slps: Grid of slopes for steepest descent
    :dt: time resolution
    :A: Grid of drainage areas
    :E: Erosion rate grid (can be input based on previous result, otherwise set to zero)
    :carea: critical area
    :return: Fluvial Erosion map

    Fluvial erosion using explicit form of transport limited eqn.  Seperated from the main class so that it can
    be parallelized
    """
    ny, nx = np.shape(slps)
    sedacc = np.zeros((ny, nx))
    f = (dx * dy) ** m

    for ij in range(len(I) - 1, 0, -1):

        i, j = lind(I[ij], ny)
        i2, j2 = lind(s[i, j], ny)
        if A[i, j] > carea:
            E[i, j] += k[i2, j2] * f * \
                np.power(A[i, j], m) * np.power(slps[i, j], n) - G * sedacc[i, j] / A[i, j]
        sedacc[i2, j2] += E[i, j]
    E *= dt

    return E

def smooth(windowSize,I,s,z,acc,athres=5):
    
    zsd = np.zeros((windowSize,len(I)))
    zsu = np.zeros((windowSize,len(I)))
    avgs=np.zeros(len(I))
    distsU=np.zeros(len(I))
    distsD = np.zeros(len(I))
    ns = np.zeros(len(I))
    amaxs = np.zeros(len(I))
    for i in range(len(I)):
        if s[i] != I[i]:
            zsd[:-1,I[i]] = zsd[1:,s[i]]
            zsd[-1,I[i]] = z[s[i]]
    for i in range(len(I)-1,0,-1):
        if s[i] != I[i]:
            if (acc[I[i]] >= windowSize) and (acc[I[i]] >= amaxs[s[i]]):
                amaxs[s[i]] = acc[I[i]]
                zsu[:,s[i]] = zsu[:,I[i]]
                zsu[:-1,s[i]] = zsu[1:,I[i]]
                zsu[-1,s[i]] = z[I[i]]
    for i in range(len(I)):
        lu = len(np.where(zsu[:,i]>0)[0])
        ld = len(np.where(zsd[:,i]>0)[0])
        minl = min([lu,ld])
        if minl>=windowSize:
            avgs[i] = np.mean(np.concatenate([zsd[:,i][zsd[:,i]>0], zsu[:,i][zsu[:,i]>0]]))
        elif minl>=1:
            avgs[i] = np.mean(np.concatenate([zsd[:,i][zsd[:,i]>0][-minl:], zsu[:,i][zsu[:,i]>0][-minl:]]))
        else:
            avgs[i] = z[i]
    return avgs



@jit(nopython=True)
def diffuse(Z, D=1.0, dy=90, dx=90, dt=1):
    """
    Explicit diffusion for hillslopes

    :param D: Diffusivity
    :param Z: Elevation
    :param dy: x resolution
    :param dt: time resolution

    """
    ny, nx = np.shape(Z)
    E = np.zeros((ny, nx))
    for i in range(1, ny - 1):
        for j in range(1, nx - 1):
            zijp = Z[i, j + 1]
            zijm = Z[i, j - 1]
            zimj = Z[i - 1, j]
            zipj = Z[i + 1, j]

            if zijp <= 0:
                zijp = Z[i, j]
            if zijm <= 0:
                zijm = Z[i, j]
            if zimj <= 0:
                zimj = Z[i, j]
            if zipj <= 0:
                zipj = Z[i, j]

            E[i, j] = D * ((2 * Z[i, j] -
                            zipj -
                            zimj) /
                           (dy ** 2) +
                           (2 * Z[i, j] - zijp - zijm)
                           / (dx ** 2))

    E *= dt

    return E


@jit(nopython=True)
def acc(I, s, init=1):
    """
    Calculate drainage area or sum some input quantity (e.g. sediment) along the stack

    :param init: Initial quantity to sum (default is ones)

    """
    ny, nx = np.shape(s)
    A = np.ones((ny, nx))

    if len(init) >= 1:
        A[:, :] = init[:, :]
    for ij in range(len(I) - 1, 0, -1):
        i, j = lind(I[ij], ny)
        i2, j2 = lind(s[i, j], ny)
        if I[ij] != s[i, j]:
            A[i2, j2] += A[i, j]
    return A


if __name__ == '__main__':
    # An example run
    A = simple_model()
    fig = plt.figure()

    for t in range(0, int(A.t / A.dt)):  # main loop
        start = timeit.default_timer()
        A.sinkfill()
        A.slp()
        A.stack()
        A.acc()
        A.erode()
        A.Z += 1
        A.Z[:, 0] = 0
        A.Z[:, -1] = 0
        A.Z[0, :] = 0
        A.Z[-1, :] = 0
        end = timeit.default_timer()

        a = plt.imshow(A.Z)
        plt.colorbar(a)

        plt.pause(.05)
        plt.clf()

    print(np.where(A.I < 1))
