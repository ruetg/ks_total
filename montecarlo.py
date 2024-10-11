from os.path import exists
import ast
import numpy as np
import sys

sys.path.insert(1, 'lem')
import lem
import multiprocess as mp
import matplotlib.pyplot as plt
import rasterio as rio 

################ -------Params-------- ##################################
class montecarlo():
    def __init__(self,nr=500, nproc=4, dy = 30.87,n_basin=1, dem_folder='./dems',basenm = '{}.tif',outfolder='./results'):
        #Basic setup
        self.dem_folder = dem_folder#'./example_data/nz_dems/'
        #name format of DEMs
        self.basenm = basenm#'nz_clipped_srtm15+_{}.tif'
        # Modify based on in/out folders
        self.outfolder = outfolder#'./results/NZ1/'
        # Number of processors
        self.n_proc = nproc
        # Number of simulations (parameter sets)
        self.nr = nr
        #dx for DEM pixels
        self.dy  = dy
        # number of basins in octopus/excel folder
        self.n_basin = n_basin
        
        ## Stream power parameters - Prior distributions
        # Vector of n values.  Currently n ranges from 0-4
        self.ns = np.random.rand(nr) * 4.0
        # The ratio D/k. Currently the prior distribution is log-uniform from 0 to 10
        self.diffus = np.power(10.0, np.random.rand(nr) * 12 + 1) * 1e-11  
        # A_crit values range as a log-uniform distribution
        self.careas = np.power(10.0, np.random.rand(nr) * 3)+1
        # diffusion exponents (p) - default is linear only (1)
        self.ps = np.zeros(nr) + 1.0
         # Vector of m values. m depends on n in most cases this is best way to optimize
        self.ms = self.ns * 0.45 

        self.Gs = np.zeros(nr)

        ## Outputs
        # The erosion rates for each basin, to contain a vector of avg erosion rates
        self.eros1 = [None] * n_basin
        # vector of avg slopes in each basin, to compare to octopus slopes
        self.slpsall = np.zeros(n_basin)



    def par_ero(self,datums):
        """
        Parallel erosion routine for each DEM basin

        :param i: basin number
        :param slps: Slopes grid calculated from preprocessing step
        :param I: The array of orderened nodes according to the Fastscape stack (1, m x n). Nodes correspond to grid locations in row-first order (order='F').
        :param s: The grid of receivers (m x n).  
        
        """
        ## The parellelized implementation of this function requires everything to be passed as 1 var.
        Zi = datums[0]
        slps = datums[1]
        A1 = datums[2]
        I = datums[3]
        s = datums[4]
        dx = datums[5]
        i = datums[6]

        # First run diffusion - we raise the coefficient by 1/p and then raise the whole
        # diffusion rate by p 

        A=A1.copy()
        


        E = lem.diffuse(
            Zi, D=-(self.diffus[i]**(1.0 / self.ps[i])), dy=self.dy, dx=dx, dt=1)
        E[E < 0] = 0
        # We only want the erosion part....
        
        E = E**self.ps[i] # Raise to p
        
        # Now add the hillslope erosion and the fluvial erosion
        ero = lem.erode_explicit(
            slps,
            I,
            s,
            A,
            E,
            dx=dx,
            dy=self.dy,
            m=self.ms[i],
            n=self.ns[i],
            k=np.zeros(np.shape(slps)) + 1e-8,
            carea=self.careas[i],G=self.Gs[i])

        # Sum erosion downstream = we do it this way so that nodes draining
        # outside of the basin (i.e. on the edge) are not included
        A = lem.acc(I, s, init=ero.copy())
        # Calculate the avg erosion per drainage area...
        pl = (A.ravel()[np.argmax(A1.ravel())]) / np.max(A1.ravel())

        return pl, i
    def preprocess(self,dem):
        lat = dem.xy(0, 0)[1]

        dx = np.cos(lat / 180 * np.pi) * self.dy  # dx is dependent on latitude
        f = lem.simple_model()
        f.dx = dx
        f.dy = self.dy

        # We must pad the DEM in order to prevent edge effects
        demz = np.float64(np.squeeze(dem.read()))
        if demz.size < 16:
            print(f'DEM is too small. Continuing.')
            return
        f.set_z(np.pad(demz, pad_width=2))

        # Outlet nodes are at or below 0
        f.BC = np.where(f.Z.transpose().ravel() <= 0)[0]

        # Fill local sinks
        f.sinkfill()

        # calculate local slopes and populate the receiver grid
        f.slp_basin()

        # Build the Fastscape stack
        f.stack()

        # calculate the receiver grid
        f.acc()

        # Get Elevation, corrected
        Zi = f.Z.copy()

        # Get drainage area
        A1 = f.A.copy()

        # Initialize mean erosion rate (per basin) vector
        mnmat = np.zeros((len(self.ms), 1))

        #Get important values from the pre-processed DEM
        A1 = f.A.copy()

        I1 = f.I.copy()
        s1 = f.s.copy()
        dy1 = f.dy
        slps = f.slps.copy()

        datums = [[Zi,slps, A1, I1, s1, dy1, n] for n in np.arange(self.nr)]
        with mp.Pool(self.n_proc) as procs:
            vals = procs.map(self.par_ero, datums)#Run the erosion model under each set of params

        eroave = mnmat[list(zip(*vals))[1], 0] = list(zip(*vals))[0] #This is where the erosion rates are saved for each trial
        
        f.acc(slps)#We also calculate the average D8 slope of the basin...
        slpave = (f.A.ravel()[f.Z.ravel() > 0][np.argmax(A1.ravel()[f.Z.ravel() > 0])]) / np.max(A1.ravel()[f.Z.ravel() > 0])
        return eroave, slpave

    def iterate(self):
        for c in range(0, self.n_basin):
            demfile = self.dem_folder + self.basenm.format(str(c))
            print(demfile)
            if exists(demfile):
                with rio.open(demfile) as dem:
                    eroave, slpave = self.preprocess(dem)
                    self.eros1[c] = eroave
                    self.slpsall[c] = slpave
            else:
                print('DEM does not exist. Continuing')

    def save(self):
        # Mean erosion rates for each model at each basin
        np.save('{}/eros'.format(self.outfolder), self.eros)
        # The array of random diffusion ratios used
        np.save('{}/diffu'.format(self.outfolder), self.diffus)
        np.save('{}/ms'.format(self.outfolder), self.ms)  # The array of random m values used
        np.save('{}/ns'.format(self.outfolder), self.ns)  # The array of random n values used
        # The array of random areas used
        np.save('{}/careas'.format(self.outfolder), self.careas)
        # Calculated avg slopes of each basin
        np.save('{}/slps2'.format(self.outfolder), self.slpsall)
        #Diffusion exponents
        np.save('{}/dns'.format(outfolder), ps)
