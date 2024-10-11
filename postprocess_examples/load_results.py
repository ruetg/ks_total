"""
This script shows how to load in some example outputs, and calculate MAE for every model run. This uses the example of a simple model with only 20 trials under constant Acrit and Diffusion, varying only n. 
"""
import matplotlib.pyplot as plt
import numpy as np
from numba import jit
import pandas as pd
import geopandas as gpd
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score as r2

def reorganize_datastructure(eros1):
    """
    Sometimes we want the data in order of model[basin] rather than basin[model]. This function reorganizes the 
    datastructure of eros1 to the latter format.

    :eros1: erosion rates exported from the model
    :exln: Length of the dataset (probably this part can be implicit... have not implemented this yet)
    
    :return: dfs, the basins in order of model [basin]
    
    """
    dfs = np.array([None]*len(eros1[1])) # Datasets organized as array[model][basin]
    c=0
    for i in range(len(dfs)): #Loop through each basin, reorganize
            dfs[i] = np.zeros(len(eros1))

            for k in range(len(eros1)):

                if eros1[k] is None:
                    continue
                dfs[i][k] = eros1[k][i]
                c+=1
    return dfs


folder = './results//nz1//' #Example result

octopus_folder = './example_data/nz_basins_/nz_basins_.shp'
a = gpd.read_file(octopus_folder)
print(a)

#a.index=a['OBSID1']
ns = np.load('{}/ns.npy'.format(folder),allow_pickle=True)
ms = np.load('{}/ms.npy'.format(folder),allow_pickle=True)
diffus = np.load('{}/diffu.npy'.format(folder),allow_pickle=True)
eros1 = np.load('{}/eros.npy'.format(folder),allow_pickle=True)
#print(a.index)
eros1 = eros1[a.index]# Use index of octopus with deleted vars

careas = np.load('{}/careas.npy'.format(folder),allow_pickle=True)
slps2 = np.load('{}/slps2.npy'.format(folder),allow_pickle=True)
dns = np.load('{}/dns.npy'.format(folder),allow_pickle=True)

## Example calculating MAE for all values
dfs = reorganize_datastructure(eros1)


L=len(eros1[1])
mae_vals = np.zeros(L) #The MAE values
r2_vals = np.zeros(L)
intercepts = np.zeros(L)
for i in range(L):#
        if(np.all(dfs[i])==None):
            continue #In some cases there will be basins we are not using, so we skip them

        y = np.float64(a['ebe_mmkyr'.upper()]) / 1000 # observed data in mm/yr

        x = dfs[i] #Modeled data for run i


        #In a few cases there will be erosion rates that are 0 in either the modeled or real data which are NaN / inf
        I = np.where(x<=0)[0] 
        x=np.delete(x,I)
        y=np.delete(y,I)
        I2 = np.where(y<=0)[0]
        x = np.delete(x,I2)
        y = np.delete(y,I2)
        
        intercept1 = np.mean(np.log(y) - np.log(x) ) #The mean of  the log transformed residuals
        
        y_est = np.log(x)+intercept1
        ####


        intercepts[i] = intercept1
        mae_vals[i] = mae(np.log(y.reshape(-1,1)),y_est.reshape(-1,1)) #Get the corcoefs
        r2_vals[i] = r2(np.log(y.reshape(-1,1)),y_est.reshape(-1,1)) #Get the corcoefs
        
## Then plot

plt.plot(ns,mae_vals,'.')
plt.xlabel('n')
plt.ylabel('MAE')
plt.show(block=True)

plt.figure()
plt.plot(ns,r2_vals,'.')
plt.xlabel('n')
plt.ylabel('R2 score')
plt.show(block=True)


plt.figure()
plt.plot(dfs[np.argmin(mae_vals)]*intercepts[np.argmin(mae_vals)],a['EBE_MMKYR'],'.')
plt.xlabel('')
plt.ylabel('')
plt.show(block=True)

plt.pause(1)


n = np.argmax()


