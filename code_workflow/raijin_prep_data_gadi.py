# This code "preps" the original Q-GCM (ocean) output by: 
#  - reorganizing the dimensions so time is accessed last
#  - saving it in chunks of 100x100 spatial chunks and in 1 year chunks 
# This reorganization of the dimensions is necessary to avoid i/o problems when accessing every timestep at a specific (x,y), which is needed to perform frequency-domain Fourier analysis.
#
# This code was run on gadi at NCI Australia.
# Code by Paige Martin and James Munroe.


import os
from glob import glob
import numpy as np
import dask
import dask.array as da
import xarray as xr

# Define path where data are stored
datadir = '/scratch/x77/pm2987/qgcm/archive/double_gyre_run/'

# Alternate method for combining netcdf files together (ended up using sorted() function below)
#ncfiles = [f for f in os.listdir(datadir) if f.startswith('output')]
#ncfiles.sort()
#ncfiles = ncfiles[200:270]
#ncfiles = [datadir + s + '/ocpo.nc' for s in ncfiles]
#print(ncfiles)

# Join the netcdf files together
ncfiles = sorted(glob(os.path.join(datadir, 'output4*/ocpo.nc'))) # this joins all output4xx/ocpo.nc files together
#ncfiles = ncfiles[273:373] # this chooses a specific subset of the above ncfiles

# Define (and create if it doesn't exist) output directory to save prepped data in
indatadir = datadir
outdatadir = '/scratch/x77/pm2987/Spunup_ocpo/' # output directory
for ncfile in ncfiles: 
    outncfile = ncfile.replace(indatadir, outdatadir) 
    outdir = os.path.dirname(outncfile)
    if not os.path.exists(outdir): # if output pathname doesn't exist, create new folder
        os.mkdir(outdir)
    if os.path.exists(outncfile):
        continue

    ds = xr.open_dataset(ncfile)\
       .isel(time=slice(1,None)) # remove first time value since it repeated at the
                                  # start of each output file   
    ds = ds.transpose('zi', 'z', 'yp', 'xp', 'time') # transpose dimensions

    # Define chunk sizes for each variable in the ocpo.nc file
    encoding = { 'p'    : {'chunksizes': [1, 100, 100, 365]},
             'q'    : {'chunksizes': [1, 100, 100, 365]},
             'h'    : {'chunksizes': [1, 100, 100, 365]},
             'taux' : {'chunksizes': [100, 100, 365]},
             'tauy' : {'chunksizes': [100, 100, 365]},              
             'e'    : {'chunksizes': [100, 100, 365]},
               }

    for v in encoding:
        encoding[v].update({'complevel': 0, 
                        'contiguous': False,
                        'dtype': np.dtype('float32'),
                        'source': ncfile,
                        'zlib': False})

    # Save newly prepped data as netcdf file
    ds.to_netcdf(outncfile, engine='netcdf4', format='NETCDF4', encoding=encoding)
    del ds
