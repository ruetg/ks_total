## 1 ) Requirements

## Install these in the following order is most likely to work

conda create --name lem_opti \
conda activate lem_opti \
conda update --all  \
conda install -c numba numba \
conda install -c conda-forge geopandas \
conda install -c conda-forge jupyterlab \
conda install -c conda-forge rasterio \
conda install -c conda-forge seaborn  

## For using the tests...
conda install -c conda-forge richdem \
conda install -c conda-forge landlab 
