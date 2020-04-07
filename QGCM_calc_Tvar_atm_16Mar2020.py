#!/usr/bin/env python
# coding: utf-8

# Run the spectral energy budget analysis code for the atmosphere

# Input arguemnts: (these input arguments are just listed after calling python3 QGCM_calc_Tvar_atm_16Mar2020.py [input1] [input2] ...)
#    1) string: 'FC','YPC','ATO' indicating which coupling of model to run analysis on
#    2) ts_length: Number of years to calculate over (e.g. 100 for 100-year timeseries)
#    3) overlap: Number of years to overlap windows (e.g. 50 to have 50-year overlap)
#    4,5) nloops: Number of first and last loop: 0 1 for first loop, 0 2 for first two loops, etc.
#    6 to end) strings: indicates which terms to calculate - options are: 'H_adv','vert_adv','Ta2','Tana','Tn1','Tdel2T','Tdel4T','TFs' 

import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

from detrend_func import detrend_func
from window_func import main as window_func
import QGCM_Tvar_atm_funcs_16Mar2020 as Q

import dask
from distributed import Client, LocalCluster
import dask.array as da

import xarray as xr
from itertools import product

import sys


# ### Define constants to be used

# In[3]:

tile_size = 100

dx = 80000. # meters
dy = 80000. # meters
dt = 1 # in days
H = [2000.0, 3000.0, 4000.0]  # meters
Htot = H[0] + H[1] + H[2]
Hm = 1000. # meters
f0 = 9.37456*(10**(-5)) #1/s (Coriolis parameter)
g = [1.2, .4]
rho = 1.0 # kg/(m^3)
Cp = 1000. # specific heat of atm in J/(kg*K)
T1 = 287
ocnorm = 1.0/(960*960) #from QGCM write-up
atnorm = 1/(384*96)
dt = 1 #86400.0 # seconds in a day (time difference between each point in time)
K2 = 2.5e4 # m^2/s (called st2d in input.params)
K4 = 2e14 # m^4/s (called st4d in input.params)
tmbaro = 3.04870e2 # mean ocean mixed layer absolute temperature (K)
tmbara = 3.05522e2 # mean atmosphere mixed layer absolute temperature (K)
toc1 = -1.78699e1 # Relative temperature for ocean layer 1

# --------------------------------
# The following are radiation constants output from the model
Bmup = -0.0213270418023 # ml eta coeffs
Dmup = 2.9649482139808 # temperature perturbation coeffs
A11down = 1.68632e-3 # QG internal interface eta coeffs
A12down = -1.63976e-3 # QG internal interface eta coeffs
B1down = 0.0116366566754 # ml eta coeffs

Dmdown = -3.2342349507402
D0up = 6.4271329356690
lamda = 35.00000 # sensible/latent heat flux coefficient (W m^2/kg) 
Fs_prime = 80.0 # perturbation amplitude of incoming solar radiation (W/m^2)

# In[4]:


def calc_terms(ts_length,overlap,nloops,start_yr,Ta2_flag,H_adv_flag,vert_adv_flag,Tdel2T_flag,Tdel4T_flag,Tana_flag,Tn1_flag,TFs_flag):

    xi = range((97+tile_size+1)//(tile_size)) 
    yi = range((385+tile_size+1)//(tile_size))
    tile_indexes = list(product(*[yi, xi]))   
 
    print(tile_indexes)
    
    # All files
    ncfiles1 = sorted(glob(os.path.join(datadir1, 'output*/atpa.nc'))) # for all atpa files
    ncfiles2 = sorted(glob(os.path.join(datadir2, 'output*/atast.nc'))) # for all atast files 
    for i in np.arange(nloops[0],nloops[1]):
        
        Ta2_sum = 0
        Tum_sum = 0
        Tvm_sum = 0
        vert_adv_sum = 0
        Tdel2T_sum = 0
        Tdel4T_sum = 0
        Tana_sum = 0
        Tn1_sum = 0
        TFs_sum = 0

        print('Slicing ncfiles')
        print((start_yr-34) + i*overlap)
        print((start_yr-34) + i*overlap + ts_length)
    
        print('i = ',i)
        # Load atpa    
        ncfiles1_loop = ncfiles1[i*overlap:i*overlap + ts_length] #ncfiles1[(i*50):(i*50+100)] # 0 starts with year 233
        print(ncfiles1_loop[0],ncfiles1_loop[-1])
        
       	chunks1 = {'xp': tile_size, 'yp': tile_size,'time':365, 'z':1, 'zi':1}
        datasets1 = [xr.open_dataset(fn,chunks=chunks1) for fn in ncfiles1_loop]
        dsx1 = xr.concat(datasets1, dim='time', coords='all')

        # Load atast
        ncfiles2_loop = ncfiles2[i*overlap:i*overlap + ts_length] #ncfiles2[(i*50):(i*50+100)] # this errors if only loa$
        print(ncfiles2_loop[0],ncfiles2_loop[-1])
        #print(ncfiles2_loop)

        chunks2 = {'xt': tile_size, 'yt': tile_size,'time':365}
        datasets2 = [xr.open_dataset(fn,chunks=chunks2) for fn in ncfiles2_loop]
        dsx2 = xr.concat(datasets2, dim='time', coords='all')
        
       	print('Finished making xarray dataArrays')
        
        if Ta2_flag:
            for tile_index in tile_indexes[:]:
                print('calculating Ta2...',tile_index)
            
                Ta2 = Q.calc_Ta2(dsx2,tile_index)
                Ta2_sum += Ta2
        
       	    np.save(save_datapath+'Ta2_'+savename+'_'+str(i*overlap+start_yr)+'_'+str(i*overlap+start_yr+ts_length-1),Ta2_sum)
            print(save_datapath+'Ta2_'+savename+'_'+str(i*overlap+start_yr)+'_'+str(i*overlap+start_yr+ts_length-1))    
       
        if Tana_flag:
            for tile_index in tile_indexes[:]:
                print('calculating ToTa...',tile_index)
            
                Tana = Q.calc_Tana(dsx2,tile_index)
                Tana_sum += Tana
            
            np.save(save_datapath+'Tana_'+savename+'_'+str(i*overlap+start_yr)+'_'+str(i*overlap+start_yr+ts_length-1),Tana_sum)
            print(save_datapath+'Tana_'+savename+'_'+str(i*overlap+start_yr)+'_'+str(i*overlap+start_yr+ts_length-1))

        if H_adv_flag:
            for tile_index in tile_indexes[:]:
                print('calculating H_adv...',tile_index)
                Tum,Tvm = Q.calc_H_adv(dsx1,dsx2,tile_index)
                Tum_sum += Tum
                Tvm_sum += Tvm            
            np.save(save_datapath+'Tum_'+savename+'_'+str(i*overlap+start_yr)+'_'+str(i*overlap+start_yr+ts_length-1),Tum_sum)
            np.save(save_datapath+'Tvm_'+savename+'_'+str(i*overlap+start_yr)+'_'+str(i*overlap+start_yr+ts_length-1),Tvm_sum)
            print(save_datapath+'Tvm_'+savename+'_'+str(i*overlap+start_yr)+'_'+str(i*overlap+start_yr+ts_length-1))
                
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

        if Tn1_flag:
            for tile_index in tile_indexes[:]:
                print('calculating Tn1...',tile_index)
                Tn1 = Q.calc_Tn1(dsx1,dsx2,tile_index)
                Tn1_sum += Tn1
            np.save(save_datapath+'Tn1_'+savename+'_'+str(i*overlap+start_yr)+'_'+str(i*overlap+start_yr+ts_length-1),Tn1_sum)
            print(save_datapath+'Tn1_'+savename+'_'+str(i*overlap+start_yr)+'_'+str(i*overlap+start_yr+ts_length-1))

        if TFs_flag:
            for tile_index in tile_indexes[:]:
                print('calculating TFs...',tile_index)
                TFs = Q.calc_TFs(dsx2,tile_index)
                TFs_sum += TFs
            np.save(save_datapath+'TFs_'+savename+'_'+str(i*overlap+start_yr)+'_'+str(i*overlap+start_yr+ts_length-1),TFs_sum)
            print(save_datapath+'TFs_'+savename+'_'+str(i*overlap+start_yr)+'_'+str(i*overlap+start_yr+ts_length-1))


        
        print('Finished the loop')


print(sys.argv)


# FC paths
if sys.argv[1] == 'FC':
    datadir1 = '/g/data/v45/pm2987/Spunup_atpa/'
    datadir2 = '/g/data/v45/pm2987/Spunup_atast/'
    datadir3 = '/g/data/v45/pm2987/Spunup_ocsst/'
    save_datapath = '/g/data/v45/pm2987/SpecTransfers_Tvar/atm/'
    savename = 'atm_Tvar_17Mar2020'
    other_name = ''
    yr_start = 34 # starting year for name saving

# YPC
elif sys.argv[1] == 'YPC':
    datadir1 = '/g/data/hh5/tmp/pm2987/Spunup_ycexp_ocpo/'
    datadir2 = '/g/data/hh5/tmp/pm2987/Spunup_ycexp_ocsst/'
    datadir3 = '/g/data/hh5/tmp/pm2987/Spunup_ycexp_Tma_oc/'
    save_datapath = '/g/data/hh5/tmp/pm2987/Tvar_specTransfers/'
    savename = 'at_ycexp_Tvar_17Mar2020'
    other_name = ''
    yr_start = 233 # starting year for name saving


ts_length = int(sys.argv[2]) # Number of years to calculate over
overlap = int(sys.argv[3]) # Number of years to overlap windows
nloops = [int(sys.argv[4]),int(sys.argv[5])] # Number of first and last loop: (0,1) for first loop, (0,2) for first two loops, etc.
print(nloops)

# Determine which term to calculate
for i in sys.argv[6:]:
    print(i)
    if i == 'Ta2':
        Ta2_flag = True
    else:
        Ta2_flag = False
    if i == 'H_adv':
        H_adv_flag = True
    else:
        H_adv_flag = False
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
    if i == 'Tana':
        Tana_flag = True
    else:
        Tana_flag = False
    if i == 'Tn1':
        Tn1_flag = True
    else:
        Tn1_flag = False
    if i == 'TFs':
        TFs_flag = True
    else:
        TFs_flag = False




if __name__ == '__main__':
    # Create a Dask Distributed cluster with one processor and as many
    # threads as cores
    #cluster = LocalCluster(n_workers=1)
    client = Client(n_workers=1,local_directory='/scratch/x77/pm2987/dask_dump/dask-worker-space')
    
    calc_terms(ts_length,overlap,nloops,yr_start,Ta2_flag,H_adv_flag,vert_adv_flag,Tdel2T_flag,Tdel4T_flag,Tana_flag,Tn1_flag,TFs_flag)
    
    




