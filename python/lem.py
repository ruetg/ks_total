import numpy as np
from dill.source import likely_import
from numba.experimental import jitclass
from numba import jit, int64, float64, bool_, int8
import math
from numba_pq import PQ as pq
import matplotlib.pyplot as plt
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
    ('BCX', int8[:, :]),
    ('slps', float64[:, :]),
    ('pour_point', int64[:]),
    ('D',float64),
    ('outlet',float64[:,:]),
    ('__dynamic_bc', bool_),
    ('a_crit', float64),
    ('U', float64[:,:]),
    ('Esum', float64[:,:]),
    ('tmp', float64[:,:]),
    ('ero', float64[:, :]),
    ('layer_depth', float64[:, :]),
    ('G2',float64),
    ('glacial_input',float64),
    ('landsurf', float64[:]),
    ('watersurf', float64[:]),
    ('useprecip', bool_),
    ('precip', float64),
    ('evaprate',float64),
    ('chi_U', float64[:,:]),
    ('G',float64)]


@jitclass(spec)
class simple_model:
    def __init__(self):
        # Model parameters
        self.m = 0.45  # Drainage area exponent
        self.G = 0 # sediment transport coeffieicnet (Yuan et al., 2019)
        self.n = 1.0
        self.dx = 1000.0  # grid spacing (m)
        self.dy = 1000.0
        self.t = 100e6  # total time (yr)
        self.dt = 1e6  # Time step
        self.__nx = 500  # Number of x grid points
        self.__ny = 500  # Number of y grid points
        self.D = 0.01
        self.layer_depth = np.zeros((self.__ny,self.__nx))
        self.G2 = .05
        self.a_crit = 0
        # Data Structures
        self.slps = np.ones((self.__ny, self.__nx), dtype=np.float64)
        self.landsurf = np.zeros((self.__ny * self.__nx), dtype=np.float64)
        self.precip = 1 #m/yr
        self.evaprate = .5 #m/yr
        self.Esum = np.zeros((self.__ny, self.__nx),dtype=np.float64)
        self.ero = np.zeros((self.__ny, self.__nx),dtype=np.float64)
        self.U = np.zeros((self.__ny, self.__nx), dtype=np.float64)
        self.__Z = np.random.rand(self.__ny, self.__nx) * 10  # Elevation
        self.receiver = np.zeros((self.__ny, self.__nx), dtype=np.int64) #Receiver grid
        self.k =  1e-6  # Erodibility
        self.k_grid = np.zeros((0 , 0))
        self.outlet = np.zeros((self.__ny,self.__nx))
        # Boundary condition grid, 0 = normal 1 = outlet
        self.BCX = np.zeros(np.shape(self.__Z), dtype=np.int8)
        self.glacial_input = 0.0
        self.BCX[:, 0] = 1  # by default set all edges as outlets
        self.BCX[:, -1] = 1
        self.BCX[0, :] = 1
        self.BCX[-1, :] = 1
        self.chi_U = np.float64([[0],[0]])
        self.ero = np.zeros(np.shape(self.__Z), dtype=np.float64)
        self.watersurf = self.__Z.transpose().ravel()
        self.stackij = np.zeros(0,dtype=np.int64)
        self.useprecip = False

        self.__dynamic_bc = False # dynamic baselevel
        # Convert the boundary condition grid to linear (for speed in some
        # cases)
        self.pour_point = np.int64([-1,-1]) #Pour_point is an (y,x) coordinate of the pour point if applicable - for drainage extraction


    def sinkfill(self):
        """
        Fill pits using the priority flood method of Barnes et al., 2014.
        """
        if self.__dynamic_bc: #We must do this at every step to ensure we have the BCs, the computational cost is low...
            self.turn_on_off_dynamic_bc(True)
        eps = 1e-6 #Minimum elevation difference between adjacent cells
        c = int(0)
        nn = self.__nx * self.__ny
        p = int(0)
        closed = np.full(nn, False)
        pit = np.zeros(nn, dtype=np.int32)
        idx = np.array([1, -1, self.__ny, -self.__ny, -self.__ny + 1, -self.__ny - 1,
               self.__ny + 1, self.__ny - 1])  # Linear indices of neighbors
        open = pq(self.__Z.transpose().flatten())
        Id1 = np.isnan(self.__Z.ravel())
        self.__Z.ravel()[Id1] = 0.0
        for i in range(self.__ny):
            for j in range(self.__nx):
                if self.BCX[i, j] == 1:
                    ij = i + j * self.__ny
                    open = open.push(ij)
                    closed[ij] = True
                c += 1
        i_shuffle = np.arange(0, self.__ny)
        j_shuffle = np.arange(0, self.__nx)
        np.random.shuffle(j_shuffle)
        np.random.shuffle(i_shuffle)
        for i in i_shuffle:
            for j in j_shuffle:
                if (i == 0) or (j == 0) or (
                        j == self.__nx - 1) or (i == self.__ny - 1):
                    # In this case only edge cells, and those below sea level
                    # (base level) are added
                    ij = j * self.__ny + i
                    if not closed[ij]:
                        closed[ij] = True
                        open = open.push(ij)
                        c += 1

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
            np.random.shuffle(idx)
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
                            self.__Z[ii, jj] = self.__Z[si, sj] + eps * np.random.rand() + eps #the e value with some randomness- we can adjust this
                            pit[p] = ij
                            p += 1
                        else:
                            open = open.push(ij)
                            c += 1
       # self.landsurf[:] = self.__Z.transpose().ravel() #keep track of the initial grid
        self.watersurf[:] = self.__Z.transpose().ravel()
        self.__Z.ravel()[Id1] = np.nan

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
        #ny, nx = np.shape(self.__Z)
        self.BCX[:,:]=0
        if self.__dynamic_bc:
            for i in range(1,self.__ny-1):
                for j in range(1,self.__nx-1):
                    if self.__Z[i,j] < 0:
                        self.BCX[i,j] = 1
                    else:
                        self.BCX[i,j] = 0

        return
    
    def set_bc(self, bc:int8[:,:]):
        """
        Set the boundary conditions

        :param bc: Boundary condition grid 1 = outlet node 0 = non-outlet. 
        Must be same size as Z

        """
        ny, nx = np.shape(bc)

        if (ny, nx) != np.shape(self.__Z):
            raise ValueError("Wrong size for Boundary Condition grid."
            " bc matrix must be same size as Z. Maybe you have not yet set Z?")
        if np.any(bc.ravel() > 0):
            self.BCX = np.zeros(np.shape(self.__Z), dtype=np.int8)
            # Have to do this loop because of unresolved type casting issues with numba and int8
            for i in range(ny):
                for j in range(nx):
                    if bc[i,j] > 0:
                        self.BCX[i,j] = 1

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
            self.U = np.zeros(np.shape(self.__Z))

            self.slps = np.zeros((ny, nx), dtype=np.float64)
            print('resetting')
            self.BCX = np.zeros((ny, nx), dtype=np.int8)
            self.BCX[:, 0] = 1
            self.BCX[:, -1] = 1
            self.BCX[0, :] = 1
            self.BCX[-1, :] = 1
            self.layer_depth = np.zeros((ny,nx), dtype=np.float64)
            self.set_bc(self.BCX)
            self.Esum = np.zeros((ny,nx),dtype=np.float64)
            self.ero = np.zeros((ny, nx), dtype=np.float64)

            self.landsurf = self.__Z.transpose().ravel()
            print('Boundary condition values have been reset')
            self.watersurf = self.landsurf.copy()
        self.__ny = ny
        self.__nx = nx
        
     
    def get_z(self):
        return self.__Z

    def slp(self):  # Calculate slope and steepest descent
        """
        D8 slopes
        """
        if self.useprecip:
            Zi = self.__Z.copy()
            c2 = 0
            for i in range(self.__nx):
                for j in range(self.__ny):
                    self.__Z[j, i] = self.watersurf[c2]
                    c2 += 1
        eps = 1e-15

        c = 0
        irand2 = np.arange(-1, 2)
        jrand2 = np.arange(-1, 2)
        self.receiver = np.zeros((self.__ny, self.__nx), dtype=np.int64)
        if self.__dynamic_bc: #We must do this at every step to ensure we have the BCs, the computational cost is low...
            self.turn_on_off_dynamic_bc(True)
            print('here1')
        irand = np.arange(0, self.__ny)
        jrand = np.arange(0, self.__nx)
        #np.random.shuffle(irand)
       # np.random.shuffle(jrand)

        for i in irand:
            for j in jrand:
                ij = j * self.__ny + i
                mxi = 0
                self.receiver[i, j] = ij

                if (0 < i < self.__ny and j > 0 and j <
                        self.__nx - 1 and i < self.__ny - 1 and not self.BCX[i,j]
                          and not(np.isnan(self.__Z[i,j]))):
                    #np.random.shuffle(irand2)
                    #np.random.shuffle(jrand2)
                    for i1 in irand2:
                        for j1 in jrand2:
                            mp = (self.__Z[i, j] - self.__Z[i + i1, j + j1]) / (np.sqrt(
                                (float(i1 * self.dy) ** 2) + float(j1 * self.dx) ** 2) + eps)  # In case slope if zero, we add eps to ensure no div by 0
                            if mp  > mxi:
                                ij2 = (j + j1) * self.__ny + i1 + i
                                mxi = mp

                                self.slps[i, j] = mp
                                self.receiver[i, j] = ij2

                    if mxi == 0:

                        #self.BCX[i, j] = 5
                        c += 1
                    if self.BCX[i,j] > 0:
                        if self.__Z[i,j] > 0:
                            print(self.__Z[i,j])


        if self.useprecip:
            self.__Z = Zi.copy()
        return c

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

    def landsed(self):


        self.landsurf[:] = self.__Z.transpose().ravel() #keep track of the initial grid
        self.sinkfill()
        ### We calculate what topology would look like with no rainfall to find sinks
        self.watersurf[:] = self.landsurf[:]
        self.slp()
        self.stack()
        nn = self.__ny * self.__nx

        maxareasinkfill = 0

        idx = np.int64([ 1, -1, -self.__ny, self.__ny, 1 + self.__ny, 1 - self.__ny , -1 + self.__ny, -1 - self.__ny])
        precip = np.ones(nn) * self.precip * 1.0
        sed = self.ero.transpose().ravel()

        tsed = 0
        nsink = 0
        if self.useprecip:
            uselandsed=3
        else:
            uselandsed=1

        next1 = np.zeros(nn,dtype=np.int64)
        sedlist = np.zeros(nn,dtype=np.int64)
        usedr = np.zeros(nn,dtype=np.bool_)
        runoff = precip
        sinkareas = np.zeros(nn)
        Z = self.__Z.transpose().ravel()

        # Update usedr array
        for i in range( self.__ny ):
            usedr[i] = True
        for i in range(nn - self.__ny, nn):
            usedr[i] = True
        for i in range(self.__ny-1, nn, self.__ny):
            usedr[i] = True
        for i in range(0, nn, self.__ny):
            usedr[i] = True

        # Build cumulative sediment from ero and stream network
        for i in range(nn-1, 0, -1):
            i1,j1 = self.lind(self.stackij[i], self.__ny)
            if self.receiver[i1,j1] != self.stackij[i]:
                sed[self.receiver[i1,j1]] += sed[self.stackij[i]]
                runoff[self.receiver[i1,j1]] += runoff[self.stackij[i]]
        # Zero out sediment for non-sink areas
        for i in range(nn):
            i1, j1 = self.lind(self.stackij[i], self.__ny)
            if self.receiver[i1,j1] != self.stackij[i]:
                sed[self.stackij[i]] = 0
                runoff[self.stackij[i]] = 0
            else:
                nsink += 1
        tcount=0
        # Process each cell
        for i in range(nn):
            i1, j1 = self.lind(self.stackij[i], self.__ny)

            if (self.receiver[i1,j1] == self.stackij[i]) and (not(usedr[self.stackij[i]])):
                nseds = [0]
                massextra = [0.0]
                massextra_precip = [0.0]
                area = [0]
                volume = [0.0]

                self.recursivesed(self.stackij[i], sed, self.landsurf, Z, idx, usedr, sedlist, nseds, area, volume, massextra, massextra_precip, uselandsed,next1,runoff)
                fact = 0
                fact2 = 0
                if massextra[0] >= volume[0]:
                    fact = 1
                elif volume[0] > 0:
                    fact = massextra[0] / volume[0]
                if massextra_precip[0] * self.dt + massextra[0] - self.evaprate * area[0] * self.dt >= volume[0]:
                    fact2 = 1
                elif massextra[0] < volume[0]:
                    fact2 = max(0.0, (massextra_precip[0] * self.dt - self.evaprate * float(area[0]) * self.dt) / (volume[0] - massextra[0]))
                if massextra_precip[0] ==0:
                    tcount+=1
               # print(massextra_precip)
                for j in range(nseds[0]):
                    i1,j1 = self.lind(sedlist[j], self.__ny)
                    if sedlist[j] > nn or sedlist[j] < 0:
                        print("Warning: bad access in recursive sedfill")
                        break
                    addsed = (Z[sedlist[j]] - self.landsurf[sedlist[j]]) * fact
                    self.landsurf[sedlist[j]] += addsed
                    self.ero[i1,j1] -= addsed
                    addprecip = (Z[sedlist[j]] - self.landsurf[sedlist[j]]) * fact2

                    self.watersurf[sedlist[j]] = self.landsurf[sedlist[j]] + addprecip
                    sinkareas[sedlist[j]] = area[0]
        # Update land surface based on sink areas
        if maxareasinkfill >= 0:
            for i in range(0, nn):
                if sinkareas[i] < maxareasinkfill:
                    self.landsurf[i] = Z[i]
        c=0
        for i in range(self.__nx):
            for j in range(self.__ny):
                self.__Z[j,i]= self.landsurf[c]
                c+=1


    def recursivesed(self,ij, sed, landsurf, Z, idx, usedr, sedlist, nseds, area, volume, massextra, massextra_precip, uselandsed,next1,runoff):
        usedr[ij] = True
        c = 0
        go = True

        while go:

            sedlist[nseds[0]] = ij
            nseds[0] += 1
            for i in range(8):

                if (landsurf[idx[i] + ij] < Z[idx[i] + ij]) and (not(usedr[idx[i] + ij])):

                    next1[c] = idx[i] + ij
                    usedr[idx[i] + ij] = True
                    c += 1

            volume[0] += (Z[ij] - landsurf[ij])
            area[0] += 1

            if uselandsed == 1:
                massextra[0] += sed[ij]
            elif uselandsed == 2:
                massextra_precip[0] += runoff[ij]
            elif uselandsed == 3:
                massextra[0] += sed[ij]
                massextra_precip[0] += runoff[ij]

            c -= 1
            if c >= -5:
                ij = next1[c]
            else:
                go = False
    def erode(self):
        """
        Erode using fastscape method
        """
        converge_thres = 1e-9 # min elevation err within 1 timestep
        converge_thres_sed = 1e-9 #self.k/np.sqrt(self.n)*self.dt/1e2 # for now this seems reasonable based on the inputs

        dA = (self.dx * self.dy) ** self.m
        if self.n == 1:
            ni = 1
        else:
            ni = 5
        k = self.k
        useKGrid = False
        if len(self.k_grid)  > 0:
            useKGrid = True
        max_iter = 1
        if self.G > 0:
            max_iter = 100 #max iterations for convergence
        sumsed = np.zeros(np.shape(self.__Z))
        sumsed2 = np.zeros(np.shape(self.__Z))
        Zi = self.__Z.copy()
        for iter in range(max_iter):
            sumsedi = sumsed.copy()
            self.__Z = Zi.copy()
            for ij in range(len(self.stackij)):
                i, j = self.lind(self.stackij[ij], self.__ny)
                i2, j2 = self.lind(self.receiver[i, j], self.__ny)
                if (i2 != i) | (j2 != j):
                    if useKGrid: #If we use variable k in a grid....
                        k = self.k_grid[i2, j2]
                    if self.A[i,j] >= self.a_crit:
                        dx = np.sqrt((float(i2 - i) * self.dy) ** 2 + (
                            float(j2 - j) * self.dx) ** 2)
                        # f = k * dA * self.A[i, j] ** self.m * self.dt * \
                        #     (self.__Z[i, j] - self.__Z[i2, j2])**(self.n - 1) / dx ** self.n
                        f2 = k * dA * self.A[i, j] ** self.m * self.dt/dx**self.n
                        x = 100
                        xl = 99999
                        ni = 1
                        c1 = self.__Z[i2,j2] - self.__Z[i,j]
                        c2 = self.G / self.A[i, j]  * (sumsed[i, j] - sumsed2[i,j])
                        c3 = self.U[i,j] * self.dt
                        c = c1 - c2 - c3 
                        while np.abs(x - xl) > converge_thres:
                            xl = x
                            if x<0:
                                x=0
                            x = x - (x + f2 * x ** self.n + c ) / \
                                (1 + self.n * f2 * x ** (self.n - 1.0 ))

                            ni += 1
                            if ni >=100:
                                print('Not Converged')
                                print(np.abs(x - xl) )
                                break
                            

                        self.__Z[i,j] = x + self.__Z[i2,j2]
                        if (self.layer_depth[i,j]>0) & (Zi[i,j] - self.__Z[i,j] > self.layer_depth[i, j]):
                            self.__Z[i,j] = Zi[i,j] - self.layer_depth[i, j]
                        sumsed2[i,j] = Zi[i,j] - self.__Z[i,j]

                        #In case their is <0 slopes
                        if self.G==0:
                            if sumsed2[i,j] < 0:
                                self.__Z[i,j] = Zi[i,j]
                                sumsed2[i, j] = 0

                        if self.__Z[i,j] <= 0:
                            self.__Z[i,j] = 0
                else:
                    self.__Z[i,j] += self.U[i2,j2]*self.dt
            
            
            sumsed = sumsed2.copy()  
            for ij in range(len(self.stackij)-1,0,-1):
                i, j = self.lind(self.stackij[ij], self.__ny)
                i2, j2 = self.lind(self.receiver[i, j], self.__ny)
                if (i2!=i) | (j2!=j):
                    sumsed[i2, j2] += sumsed[i, j]#np.abs(sumsed[i, j])/2 + sumsed[i, j]/2
            diffsed = np.mean(np.abs(sumsed - sumsedi))
            
            if diffsed < converge_thres_sed: # for now this seems a decent dynamic threshold...
                break
        if (self.G>0) & (iter >= max_iter-1):
            print('Not Converged')
        self.ero = Zi - self.__Z  + self.U * self.dt
        self.Esum +=  self.ero
        self.Esum[:,0]=0
        self.Esum[:,-1]=0
        self.Esum[-1,:]=0
        self.Esum[0,:]=0
        return sumsed

    def erode_explicit(self):
        """
        Erode using explicit method

        :returns: erosion rate grid
        """
        E = np.zeros((self.__ny, self.__nx))
        k = self.k
        useKGrid = False
        if len(self.k_grid > 0):
            useKGrid = True
        sumseds = np.zeros((self.__ny, self.__nx))
        for ij in range(len(self.stackij) -1,0 ,-1):

            i, j = self.lind(self.stackij[ij], self.__ny)
            i2, j2 = self.lind(self.receiver[i, j], self.__ny)
            if  useKGrid: #If we use variable k in a grid...
                k = self.k_grid[i2, j2]
            if (i2 != i) | (j2 != j):
                if self.A[i, j] > self.a_crit:
                    f = self.dt * (self.dx * self.dy) ** self.m
                    E[i, j] = k * f * \
                       self.A[i, j] ** self.m * self.slps[i, j]** self.n \
                        - self.G/self.A[i,j] * sumseds[i,j]
                sumseds[i2,j2] += sumseds[i,j] + E[i, j]
        I = (E.ravel()>self.layer_depth.ravel()) & (self.layer_depth.ravel()>1)
        E.ravel()[I] = self.layer_depth.ravel()[I]
        self.Esum += E
        self.ero = E
        self.__Z -= E 
        self.__Z += self.U * self.dt

        return sumseds

    def chicalc(self):
        """
        "params: U1 = normalized uplift rate to be included in chi calculations"
        "params: elev_fact = elevation factor for rivers that do not start at zero elevation -  Giachetta and Willett report this as 1/32.2"
        Calculate chi based on the inputs
        """
        elev_fact = 0.0
        if len(self.chi_U) > 1:
            U1 = self.chi_U
        else:
            U1 = np.ones_like(self.__Z)
        self.chi = np.zeros((self.__ny, self.__nx), dtype=np.float64)
        dA = (self.dx * self.dy) ** self.m

        for ij in range(len(self.stackij)):
            i, j = self.lind(self.stackij[ij], self.__ny)
            i2, j2 = self.lind(self.receiver[i,j], self.__ny)
            ds = np.sqrt(((i - i2) * self.dy)**2 + ((j - j2) * self.dx)**2)
            if (self.receiver[i2, j2] == self.receiver[i,j]):
                self.chi[i, j] = self.__Z[i, j] * elev_fact
            else:
                self.chi[i, j] = U1[i,j] / (self.A[i, j] ** self.m * dA) * ds
                self.chi[i, j] += self.chi[i2, j2]
        return

    def diffusion(self):
        """
        Explicit diffusion for hillslopes and maybe subsurface processes

        :param D: Diffusivity

        """
        Z = self.__Z
        courant_t = min(np.array([self.dx**2, self.dy**2])) / (4 * self.D)
        ny, nx = np.shape(Z)
        E = np.zeros((ny, nx))
        t_tot = 0

        while t_tot < self.dt:
            print('t')
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
            #self.ero += E
        return E

    def erode_sklar_experimental(self):
        """
        Erode using fastscape method
        """
        """
             Erode using explicit method

             :returns: erosion rate grid
             """
        E = np.zeros((self.__ny, self.__nx))
        k = self.k
        useKGrid = False
        if len(self.k_grid > 0):
            useKGrid = True
        sumseds = np.zeros((self.__ny, self.__nx))
        f = self.dt * (self.dx * self.dy) ** self.m
        es = np.zeros((self.__ny, self.__nx))
        self.tmp = np.zeros((self.__ny, self.__nx))

        for ij in range(len(self.stackij) - 1, 0, -1):

            i, j = self.lind(self.stackij[ij], self.__ny)
            i2, j2 = self.lind(self.receiver[i, j], self.__ny)
            if useKGrid:  # If we use variable k in a grid...
                k = self.k_grid[i2, j2]
            if (i2 != i) | (j2 != j):
                e = k * f * self.A[i, j] ** self.m * self.slps[i, j] ** self.n
                q =  self.G / self.A[i, j] * sumseds[i, j]
                if (q > e / 2) & (q < e):
                    E[i, j] = 2 * (e - q)
                elif q > e:

                    E[i, j] = e - q
                else:
                    E[i, j] =  (2 - self.G2) * q
                if 1:#self.A[i,j] == 1:
                    E[i,j] += self.G2 * e / 2
                    if self.glacial_input>0:
                        if self.A[i,j] == 1:
                            E[i,j] += self.glacial_input
                es[i,j] = e
               # sumseds[i, j] += E[i, j]
                sumseds[i2, j2] += E[i,j] #sumseds[i, j]
            self.Esum[i,j] = sumseds[i, j] #/(self.A+1e-10)
        self.tmp = es.copy()
        self.ero = E
        self.__Z -= E
        self.__Z += self.U * self.dt

        return sumseds