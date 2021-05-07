#!/usr/bin/env python
# coding: utf-8

# Run the spectral energy budget analysis code for the ocean

# Input arguemnts: (these input arguments are just listed after calling python3 QGCM_calc_atm_spec_analysis_15Jan2020.py [input1] [input2] ...)
#    1) string: 'FC','YPC' indicating which coupling of model to run analysis on (fully coupled or partically coupled, respectively)
#    2) ts_length: Number of years to calculate over (e.g. 100 for 100-year timeseries)
#    3) overlap: Number of years to overlap windows (e.g. 50 to have 50-year overlap)
#    4,5) nloops: Number of first and last loop: 0 1 for first loop, 0 2 for first two loops, etc.
#    6 to end) strings: indicates which terms to calculate - options are: 'KE1','KE2','KE3','PE1','PE2','Bcy','Win' (need only write as many terms as I want to calculate here - anywhere between 1 and 7)

# This code was run on the gadi supercomuter at NCI Australia.
# Written by Paige Martin

import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

from detrend_func import detrend_func
from window_func import main as window_func
import QGCM_Tvar_funcs_28Nov2019 as Q

import dask
from distributed import Client, LocalCluster
import dask.array as da

import xarray as xr
from itertools import product

import sys


# ### Define constants to be used

tile_size = 100
Hm = 100 # meters

dx = 5000. # meters
dy = 5000. # meters
dt = 1 # in days
H = [350.0, 750.0, 2900.0]  # meters
Htot = H[0] + H[1] + H[2]
Hm = 100 # meters (mixed layer height)
f0 = 9.37456*(10**(-5)) #1/s (Coriolis parameter)
g = [.015, .0075]
delta_ek = 2. # meters
alpha_bc = 0.2 # nondimensional mixed boundary condition for OCEAN
gamma = 0.5*alpha_bc + 1 # nondimensional number for ease of calculation
rho = 1000.0 # kg/(m^3)
Cp = 4000. # specific heat of ocean in J/(kg*K)
T1 = 287
ocnorm = 1.0/(960*960) #from QGCM write-up
K2 = 200 # m^2/s (called st2d in input.params)
K4 = 2e9 # m^4/s (called st4d in input.params)
tmbaro = 3.04870e2 # mean ocean mixed layer absolute temperature (K)
tmbara = 3.05522e2 # mean atmosphere mixed layer absolute temperature (K)
toc1 = -1.78699e1 # Relative temperature for ocean layer 1
mask = 2 # number of cells to mask at boundaries

# --------------------------------
# The following are radiation constants output from the model
Dmdown = -3.2342349507402
D0up = 6.4271329356690
lamda = 35.00000 # sensible/latent heat flux coefficient (W m^2/kg)
Fs_prime = 80.0 # perturbation amplitude of incoming solar radiation (W/m^2)


# In[4]:

# This function calls each individual temperature variance budget computation defined in QGCM_Tvar_funcs_28Nov2019.ipynb (dubbed "Q" below)
def calc_terms(ts_length,overlap,nloops,start_yr,To2_flag,ToTa_flag,Fme_flag,slh_flag,H_adv_flag,H_Ek_flag,H_geo_flag,vert_adv_flag,Tdel2T_flag,Tdel4T_flag):

    # Define indices for each 100x100 tile (or chunk)
    #ceil(x/y) = (x+y+1)//y
    yi = range((961+tile_size+1)//(tile_size)) 
    xi = range((961+tile_size+1)//(tile_size))

    tile_indexes = list(product(*[yi, xi]))
    
    print(tile_indexes)
    print(yi)
    
    # All files
    ncfiles1 = sorted(glob(os.path.join(datadir1, 'output*/ocpo.nc'))) # for all ocpo files
    ncfiles2 = sorted(glob(os.path.join(datadir2, 'output*/ocsst.nc'))) # for all ocsst files
    ncfiles3 = sorted(glob(os.path.join(datadir3, 'output*/Tma_oc.nc'))) # for Tma on ocean grid files    
    
    # i represents the number of each window (so i goes from 0 to 6)
    for i in np.arange(nloops[0],nloops[1]):
        
        To2_sum = 0
        ToTa_sum = 0
        Fme_sum = 0
        slh_sum = 0
        Tum_sum = 0
        Tvm_sum = 0
        vert_adv_sum = 0
        Tdel2T_sum = 0
        Tdel4T_sum = 0

        print('Slicing ncfiles')
        print((start_yr-34) + i*overlap)
        print((start_yr-34) + i*overlap + ts_length)
    
        print('i = ',i)
        
        # Select desired files    
        ncfiles1_loop = ncfiles1[i*overlap:i*overlap + ts_length] # usually this is equivalent to ncfiles1[(i*50):(i*50+100)], where overlap is 50 years and ts_length is 100 years 
        print(ncfiles1_loop[0],ncfiles1_loop[-1])
        
        # Set xarray chunking
       	chunks1 = {'xp': tile_size, 'yp': tile_size,'time':365, 'z':1, 'zi':1}
        datasets1 = [xr.open_dataset(fn,chunks=chunks1) for fn in ncfiles1_loop]
        dsx1 = xr.concat(datasets1, dim='time', coords='all')

        # Load ocsst (ocsst contains surface variables, such as SST and wekt)
        ncfiles2_loop = ncfiles2[i*overlap:i*overlap + ts_length] 
        print(ncfiles2_loop[0],ncfiles2_loop[-1])
        #print(ncfiles2_loop)

        chunks2 = {'xt': tile_size, 'yt': tile_size,'time':365}
        datasets2 = [xr.open_dataset(fn,chunks=chunks2) for fn in ncfiles2_loop]
        dsx2 = xr.concat(datasets2, dim='time', coords='all')
        
       	print('Finished making xarray dataArrays')
        
        if To2_flag:
            for tile_index in tile_indexes[:]:
                print('calculating To2...',tile_index)
            
                To2 = Q.calc_To2(dsx2,tile_index)
                To2_sum += To2
        
       	    np.save(save_datapath+'To2_'+savename+'_'+str(i*overlap+start_yr)+'_'+str(i*overlap+start_yr+ts_length-1),To2_sum)
            print(save_datapath+'To2_'+savename+'_'+str(i*overlap+start_yr)+'_'+str(i*overlap+start_yr+ts_length-1))    
       
        if ToTa_flag:
            # Load atast
            ncfiles3_loop = ncfiles3[i*overlap:i*overlap + ts_length] #ncfiles2[(i*50):(i*50+100)] # this errors if only loa$
            print(ncfiles3_loop[0],ncfiles3_loop[-1])
        
            chunks3 = {'xt': tile_size, 'yt': tile_size,'time':365}
            datasets3 = [xr.open_dataset(fn,chunks=chunks3) for fn in ncfiles3_loop]
            dsx3 = xr.concat(datasets3, dim='time', coords='all')

            for tile_index in tile_indexes[:]:
                print('calculating ToTa...',tile_index)
            
                ToTa = Q.calc_ToTa(dsx2,dsx3,tile_index)
                ToTa_sum += ToTa
            
            np.save(save_datapath+'ToTa_'+savename+'_'+str(i*overlap+start_yr)+'_'+str(i*overlap+start_yr+ts_length-1),ToTa_sum)
            print(save_datapath+'ToTa_'+savename+'_'+str(i*overlap+start_yr)+'_'+str(i*overlap+start_yr+ts_length-1))

        if Fme_flag:
            for tile_index in tile_indexes[:]:
                print('calculating Fme...',tile_index)
        
                Fme = Q.calc_Fme(dsx2,tile_index)
                Fme_sum += Fme
        
            np.save(save_datapath+'Fme_'+savename+'_'+str(i*overlap+start_yr)+'_'+str(i*overlap+start_yr+ts_length-1),Fme_sum)
            print(save_datapath+'Fme_'+savename+'_'+str(i+start_yr)+'_'+str(i+100-1+start_yr))

        if slh_flag:
            for tile_index in tile_indexes[:]:
                print('calculating slh...',tile_index)
            
                slh = Q.calc_slh(dsx2,dsx3,tile_index)
                slh_sum += slh
            
            np.save(save_datapath+'slh_'+savename+'_'+str(i*overlap+start_yr)+'_'+str(i*overlap+start_yr+ts_length-1),slh_sum)
            print(save_datapath+'slh_'+savename+'_'+str(i*overlap+start_yr)+'_'+str(i*overlap+start_yr+ts_length-1))

        if H_adv_flag:
            for tile_index in tile_indexes[:]:
                print('calculating H_adv...',tile_index)
                Tum,Tvm = Q.calc_H_adv(dsx1,dsx2,tile_index,start_yr)
                Tum_sum += Tum
                Tvm_sum += Tvm            
            np.save(save_datapath+'Tum_'+savename+'_'+str(i*overlap+start_yr)+'_'+str(i*overlap+start_yr+ts_length-1),Tum_sum)
            np.save(save_datapath+'Tvm_'+savename+'_'+str(i*overlap+start_yr)+'_'+str(i*overlap+start_yr+ts_length-1),Tvm_sum)
            print(save_datapath+'Tvm_'+savename+'_'+str(i*overlap+start_yr)+'_'+str(i*overlap+start_yr+ts_length-1))
            
        if H_Ek_flag:
            for tile_index in tile_indexes[:]:
                print('calculating H_adv...',tile_index)
                Tum,Tvm = Q.calc_H_adv(dsx1,dsx2,tile_index,'Ek')
                Tum_sum += Tum
                Tvm_sum += Tvm            
            np.save(save_datapath+'Tum_Ek_'+savename+'_'+str(i*overlap+start_yr)+'_'+str(i*overlap+start_yr+ts_length-1),Tum_sum)
            np.save(save_datapath+'Tvm_Ek_'+savename+'_'+str(i*overlap+start_yr)+'_'+str(i*overlap+start_yr+ts_length-1),Tvm_sum)
            print(save_datapath+'Tvm_Ek_'+savename+'_'+str(i*overlap+start_yr)+'_'+str(i*overlap+start_yr+ts_length-1))
            
        if H_geo_flag:
            for tile_index in tile_indexes[:]:
                print('calculating H_adv...',tile_index)
                Tum,Tvm = Q.calc_H_adv(dsx1,dsx2,tile_index,'geo')
                Tum_sum += Tum
                Tvm_sum += Tvm            
            np.save(save_datapath+'Tum_geo_'+savename+'_'+str(i*overlap+start_yr)+'_'+str(i*overlap+start_yr+ts_length-1),Tum_sum)
            np.save(save_datapath+'Tvm_geo_'+savename+'_'+str(i*overlap+start_yr)+'_'+str(i*overlap+start_yr+ts_length-1),Tvm_sum)
            print(save_datapath+'Tvm_geo_'+savename+'_'+str(i*overlap+start_yr)+'_'+str(i*overlap+start_yr+ts_length-1))
                
        if vert_adv_flag:
            for tile_index in tile_indexes[:]:
                print('calculating vert_adv...',tile_index)
                vert_adv = Q.calc_vert_adv(dsx2,tile_index)
                vert_adv_sum += vert_adv
            np.save(save_datapath+'vert_adv_'+savename+'_'+str(i*overlap+start_yr)+'_'+str(i*overlap+start_yr+ts_length-1),vert_adv_sum)
            print(save_datapath+'vert_adv_'+savename+'_'+str(i*overlap+start_yr)+'_'+str(i*overlap+start_yr+ts_length-1))
                
        if Tdel2T_flag:
            for tile_index in tile_indexes[:]:
                print('calculating Tdel2T...',tile_index)
                Tdel2T = Q.calc_Tdel2T(dsx2,tile_index)
                Tdel2T_sum += Tdel2T
            np.save(save_datapath+'Tdel2T_'+savename+'_'+str(i*overlap+start_yr)+'_'+str(i*overlap+start_yr+ts_length-1),Tdel2T_sum)
            print(save_datapath+'Tdel2T_'+savename+'_'+str(i*overlap+start_yr)+'_'+str(i*overlap+start_yr+ts_length-1))

        if Tdel4T_flag:
            for tile_index in tile_indexes[:]:
                print('calculating Tdel4T...',tile_index)
                Tdel4T = Q.calc_Tdel4T(dsx2,tile_index)
                Tdel4T_sum += Tdel4T
            np.save(save_datapath+'Tdel4T_'+savename+'_'+str(i*overlap+start_yr)+'_'+str(i*overlap+start_yr+ts_length-1),Tdel4T_sum)
            print(save_datapath+'Tdel4T_'+savename+'_'+str(i*overlap+start_yr)+'_'+str(i*overlap+start_yr+ts_length-1))

        
        print('Finished the loop')


print(sys.argv)


# FC paths
if sys.argv[1] == 'FC':
    datadir1 = '/g/data/v45/pm2987/Spunup_ocpo/'
    datadir2 = '/g/data/v45/pm2987/Spunup_ocsst/'
    datadir3 = '/g/data/v45/pm2987/Spunup_Tma_oc_temp/'
    save_datapath = '/g/data/v45/pm2987/SpecTransfers_Tvar/'
    savename = 'oc_Tvar_21Apr2020'
    other_name = ''
    yr_start = 34 # starting year for name saving

# YPC paths
elif sys.argv[1] == 'YPC':
    datadir1 = '/g/data/hh5/tmp/pm2987/Spunup_ycexp_ocpo/'
    datadir2 = '/g/data/hh5/tmp/pm2987/Spunup_ycexp_ocsst/'
    datadir3 = '/g/data/hh5/tmp/pm2987/Spunup_ycexp_Tma_oc/'
    save_datapath = '/g/data/hh5/tmp/pm2987/Tvar_specTransfers/'
    savename = 'oc_ycexp_21Apr2020'
    other_name = ''
    yr_start = 233 # starting year for name saving

# Get quantities based on user input
ts_length = int(sys.argv[2]) # Number of years to calculate over
overlap = int(sys.argv[3]) # Number of years to overlap windows
nloops = [int(sys.argv[4]),int(sys.argv[5])] # Number of first and last loop: (0,1) for first loop, (0,2) for first two loops, etc.
print(nloops)

# Set flags for the terms to be computed based on user input
for i in sys.argv[6:]:
    print(i)
    if i == 'To2':
        To2_flag = True
    else:
        To2_flag = False
    if i == 'ToTa':
        ToTa_flag = True
    else:
        ToTa_flag = False
    if i == 'H_adv':
        H_adv_flag = True
    else:
        H_adv_flag = False
    if i == 'H_adv_Ek':
        H_Ek_flag = True
    else:
        H_Ek_flag = False
    if i == 'H_adv_geo':
        H_geo_flag = True
    else:
        H_geo_flag = False
    if i == 'slh':
        slh_flag = True
    else:
        slh_flag = False
    if i == 'Fme':
        Fme_flag = True
    else:
        Fme_flag = False
    if i == 'vert_adv':
        vert_adv_flag = True
    else:
        vert_adv_flag = False
    if i == 'Tdel2T':
        Tdel2T_flag = True
    else:
        Tdel2T_flag = False
    if i == 'Tdel4T':
        Tdel4T_flag = True
    else:
        Tdel4T_flag = False





# This if statement is needed for dask to run from a python script
if __name__ == '__main__':
    # Create a Dask Distributed cluster with one processor and as many
    # threads as cores
    #cluster = LocalCluster(n_workers=4)
    #client = Client(cluster)
    
    # Ask for 8 workers, send the intermediate output (which sometimes occurs, sometimes doesn't) to PBS_JOBFS/dask-worker-space
    #  - if we don't include the dask-worker-space directory, it is automatically saved in my home folder and often results in memory issues
    client = Client(n_workers=8,local_directory=os.path.join(os.environ['PBS_JOBFS'],'dask-worker-space'))

    calc_terms(ts_length,overlap,nloops,yr_start,To2_flag,ToTa_flag,Fme_flag,slh_flag,H_adv_flag,H_Ek_flag,H_geo_flag,vert_adv_flag,Tdel2T_flag,Tdel4T_flag)
    
    




