#!/usr/bin/python
# coding: utf-8

# This code contains functions to compute each term in the frequency-domain temperature variance budget of Q-GCM (Quasi-Geostrophic Coupled Model)
# Run on the gadi supercomputer at NCI Australia.
# Written by Paige Martin

import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

from detrend_func import detrend_func
from window_func import main as window_func

import dask
from distributed import Client, LocalCluster
import dask.array as da

import xarray as xr

print('Libraries loaded')

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

# Compute horizontal advection
def calc_H_adv(dsx1, dsx2, tile_index,sep, z=0): 
    
    # dsx1 is ocpo
    # dsx2 is ocsst
    
    # Select variable
    T = dsx2.sst
    p = dsx1.p.isel(z=0)
    taux = dsx1.taux
    tauy = dsx1.tauy

    # Select specified tile, with 2-cell padding for taking of derivatives
    ix, iy = tile_index
    T = T.isel(yt=slice(max(iy*tile_size-1,0), (iy+1)*tile_size+1),
               xt=slice(max(ix*tile_size-1,0), (ix+1)*tile_size+1))
    p = p.isel(yp=slice(max(iy*tile_size-1,0), (iy+1)*tile_size+2), # +2 because p and tau on p-grid, T on T-grid
               xp=slice(max(ix*tile_size-1,0), (ix+1)*tile_size+2))
    taux = taux.isel(yp=slice(max(iy*tile_size-1,0), (iy+1)*tile_size+2),
                 xp=slice(max(ix*tile_size-1,0), (ix+1)*tile_size+2))
    tauy = tauy.isel(yp=slice(max(iy*tile_size-1,0), (iy+1)*tile_size+2),
                 xp=slice(max(ix*tile_size-1,0), (ix+1)*tile_size+2))
    xt = T.xt.values # grab values of dask arrays
    yt = T.yt.values
    time = T.time.values
    
    T = T.data
    p = p.data
    taux = taux.data
    tauy = tauy.data
    print('T and p shape with buffer of 2 on either side = ',T.shape,p.shape)
    
    # Average p and tau dimensions to match T-grid
    p = 0.5 * (p[1:,:,:]+p[:-1,:,:])
    p = 0.5 * (p[:,1:,:]+p[:,:-1,:])
    #tauy_avg = 0.25 * (tauy[1:,1:,:] + tauy[:-1,:-1,:])
    #taux_avg = 0.25 * (taux[1:,1:,:] + taux[:-1,:-1,:])
    tauy_avg = 0.5 * (tauy[1:,:,:] + tauy[:-1,:,:])
    tauy_avg = 0.5 * (tauy_avg[:,1:,:] + tauy_avg[:,:-1,:])
    taux_avg = 0.5 * (taux[1:,:,:] + taux[:-1,:,:])
    taux_avg = 0.5 * (taux_avg[:,1:,:] + taux_avg[:,:-1,:])
    print('p and tauy_avg.shape = ',p.shape,tauy_avg.shape)
    
    # Update size variables
    ny, nx, nt = T.shape
    T = T.rechunk(chunks={0: ny, 1: nx})
    p = p.rechunk(chunks={0: ny, 1: nx})
    taux_avg = taux_avg.rechunk(chunks={0: ny, 1: nx})
    tauy_avg = tauy_avg.rechunk(chunks={0: ny, 1: nx})
    
    # Take a single derivative using centered diff
    def Derivative(var, axis):
        var_right = np.roll(var, 1, axis=axis)
        var_left = np.roll(var, -1, axis=axis)
        dvar = var_left - var_right
        return dvar
        
    Derivative_x = lambda p: Derivative(p, 1)
    Derivative_y = lambda p: Derivative(p, 0)
    
    

    # Define ml velocities
    if sep == 233:
        print('running ycexp')
        um = (1./(f0*Hm)) * tauy_avg
        vm = - (1./(f0*Hm)) * taux_avg
    elif sep == 'Ek':
        print('running Ekman')
        um = (1./(f0*Hm)) * tauy_avg
        vm = - (1./(f0*Hm)) * taux_avg
    elif sep == 'geo':
        print('running geo')
        # Evaluate the derivative function
        dpdx = (1./(2*dx)) * p.map_blocks(Derivative_x)
        dpdy = (1./(2*dx)) * p.map_blocks(Derivative_y)
        um = -(1./f0) * dpdy
        vm = (1./f0) * dpdx
    else:
        # Evaluate the derivative function
        dpdx = (1./(2*dx)) * p.map_blocks(Derivative_x)
        dpdy = (1./(2*dx)) * p.map_blocks(Derivative_y)
        um = -(1./f0) * dpdy + (1./(f0*Hm)) * tauy_avg
        vm = (1./f0) * dpdx - (1./(f0*Hm)) * taux_avg
        
    
    
    # Velocity * Temp
    umTm = um * T
    vmTm = vm * T
    
    # Take derivatives of these terms
    umTm_x = (1./(2*dx)) * umTm.map_blocks(Derivative_x)
    vmTm_y = (1./(2*dx)) * vmTm.map_blocks(Derivative_y)
    
    print('umTm_x shape = ',umTm_x.shape)
    
    # Trim arrays of extra ends
    umTm_x = umTm_x[1:-1, 1:-1, :]
    vmTm_y = vmTm_y[1:-1, 1:-1, :]
    T = T[1:-1, 1:-1, :]
    ## Mask an extra cell (2 total at boundaries)
    #if ix == 0:
    #    umTm_x = umTm_x[:, 1:, :]
    #    vmTm_y = vmTm_y[:, 1:, :]
    #    T = T[:, 1:, :]
    #if ix == 9:
    #    umTm_x = umTm_x[:, :-1, :]
    #    vmTm_y = vmTm_y[:, :-1, :]
    #    T = T[:, :-1, :]
    #if iy == 0:
    #    umTm_x = umTm_x[1:, :, :]
    #    vmTm_y = vmTm_y[1:, :, :]
    #    T = T[1:, :, :]
    #if iy == 9:
    #    umTm_x = umTm_x[:-1, :, :]
    #    vmTm_y = vmTm_y[:-1, :, :]
    #    T = T[:-1, :, :]
        
   
        
    print('after masking 2 cells at boundary ',T.shape,umTm_x.shape)
    
    # Use smaller spatial chunks for fft
    T = T.rechunk(chunks={0: tile_size/10, 1: tile_size/10, 2: nt})
    umTm_x = umTm_x.rechunk(chunks={0: tile_size/10, 1: tile_size/10, 2: nt})
    vmTm_y = vmTm_y.rechunk(chunks={0: tile_size/10, 1: tile_size/10, 2: nt})
    
    print('p shape after chunking/10 = ',T.shape)
        
    # Function that detrends, windows, and takes fft
    def fft_block(var):
        var = detrend_func(var,'time')
        var = window_func(var,'time')
        varhat = (1./var.shape[2])*np.fft.rfft(var, axis=2)
        return varhat
    
    # Execute fft function above
    That = T.map_blocks(fft_block)
    umTm_xhat = umTm_x.map_blocks(fft_block)
    vmTm_yhat = vmTm_y.map_blocks(fft_block)
    
    That = That.rechunk(chunks={0: tile_size, 1:tile_size, 2: 365})
    umTm_xhat = umTm_xhat.rechunk(chunks={0: tile_size, 1:tile_size, 2: 365})
    vmTm_yhat = vmTm_yhat.rechunk(chunks={0: tile_size, 1:tile_size, 2: 365})

    # Multiply by constants
    TumTm_x = ((That.conj()*umTm_xhat).real)
    TvmTm_y = ((That.conj()*vmTm_yhat).real)
    
    print('final shape = ',TumTm_x.shape)

    # Sum over x- and y-axes
    TumTm_x = da.sum(TumTm_x, axis=(0,1))
    TvmTm_y = da.sum(TvmTm_y, axis=(0,1))
    
    #H_adv = H_adv.compute()
    TumTm_x = TumTm_x.compute()
    TvmTm_y = TvmTm_y.compute()
    
    # Wrap as DataArray
    n = len(time)
    d = time[1] - time[0]
    freq = np.fft.rfftfreq(n, d)
    
    TumTm_x = xr.DataArray(TumTm_x, 
                      dims=['freq'], 
                      coords={'freq': freq,
                              'xt': xt.mean(),
                              'yt': yt.mean()})
    
    TvmTm_y = xr.DataArray(TvmTm_y, 
                      dims=['freq'], 
                      coords={'freq': freq,
                              'xt': xt.mean(),
                              'yt': yt.mean()})
    
    return TumTm_x,TvmTm_y

# Compute x- and y-components of horizontal advection separately
def calc_H_adv_sep(dsx1, dsx2, tile_index,sep, z=0): 
    
    # dsx1 is ocpo
    # dsx2 is ocsst
    
    # Select variable
    T = dsx2.sst
    p = dsx1.p.isel(z=0)
    taux = dsx1.taux
    tauy = dsx1.tauy

    # Select specified tile, with 2-cell padding for taking of derivatives
    ix, iy = tile_index
    T = T.isel(yt=slice(max(iy*tile_size-1,0), (iy+1)*tile_size+1),
               xt=slice(max(ix*tile_size-1,0), (ix+1)*tile_size+1))
    p = p.isel(yp=slice(max(iy*tile_size-1,0), (iy+1)*tile_size+2), # +2 because p and tau on p-grid, T on T-grid
               xp=slice(max(ix*tile_size-1,0), (ix+1)*tile_size+2))
    taux = taux.isel(yp=slice(max(iy*tile_size-1,0), (iy+1)*tile_size+2),
                 xp=slice(max(ix*tile_size-1,0), (ix+1)*tile_size+2))
    tauy = tauy.isel(yp=slice(max(iy*tile_size-1,0), (iy+1)*tile_size+2),
                 xp=slice(max(ix*tile_size-1,0), (ix+1)*tile_size+2))
    xt = T.xt.values # grab values of dask arrays
    yt = T.yt.values
    time = T.time.values
    
    T = T.data
    p = p.data
    taux = taux.data
    tauy = tauy.data
    print('T and p shape with buffer of 2 on either side = ',T.shape,p.shape)
    
    # Average p and tau dimensions to match T-grid
    p = 0.5 * (p[1:,:,:]+p[:-1,:,:])
    p = 0.5 * (p[:,1:,:]+p[:,:-1,:])
    #tauy_avg = 0.25 * (tauy[1:,1:,:] + tauy[:-1,:-1,:])
    #taux_avg = 0.25 * (taux[1:,1:,:] + taux[:-1,:-1,:])
    tauy_avg = 0.5 * (tauy[1:,:,:] + tauy[:-1,:,:])
    tauy_avg = 0.5 * (tauy_avg[:,1:,:] + tauy_avg[:,:-1,:])
    taux_avg = 0.5 * (taux[1:,:,:] + taux[:-1,:,:])
    taux_avg = 0.5 * (taux_avg[:,1:,:] + taux_avg[:,:-1,:])
    print('p and tauy_avg.shape = ',p.shape,tauy_avg.shape)
    
    # Update size variables
    ny, nx, nt = T.shape
    T = T.rechunk(chunks={0: ny, 1: nx})
    p = p.rechunk(chunks={0: ny, 1: nx})
    taux_avg = taux_avg.rechunk(chunks={0: ny, 1: nx})
    tauy_avg = tauy_avg.rechunk(chunks={0: ny, 1: nx})
    
    # Take a single derivative using centered diff
    def Derivative(var, axis):
        var_right = np.roll(var, 1, axis=axis)
        var_left = np.roll(var, -1, axis=axis)
        dvar = var_left - var_right
        return dvar
        
    Derivative_x = lambda p: Derivative(p, 1)
    Derivative_y = lambda p: Derivative(p, 0)
    
    # Evaluate the derivative function
    dpdx = (1./(2*dx)) * p.map_blocks(Derivative_x)
    dpdy = (1./(2*dx)) * p.map_blocks(Derivative_y)

    # Define ml velocities
    if sep == 'Ek':
        print('running ycexp')
        um = (1./(f0*Hm)) * tauy_avg
        vm = - (1./(f0*Hm)) * taux_avg
    elif sep == 'geo':
        um = -(1./f0) * dpdy
        vm = (1./f0) * dpdx
    
    # Velocity * Temp
    umTm = um * T
    vmTm = vm * T
    
    # Take derivatives of these terms
    umTm_x = (1./(2*dx)) * umTm.map_blocks(Derivative_x)
    vmTm_y = (1./(2*dx)) * vmTm.map_blocks(Derivative_y)
    
    print('umTm_x shape = ',umTm_x.shape)
    
    # Trim arrays of extra ends
    umTm_x = umTm_x[1:-1, 1:-1, :]
    vmTm_y = vmTm_y[1:-1, 1:-1, :]
    T = T[1:-1, 1:-1, :]
    print('after masking 2 cells at boundary ',T.shape,umTm_x.shape)
    
    # Use smaller spatial chunks for fft
    T = T.rechunk(chunks={0: tile_size/10, 1: tile_size/10, 2: nt})
    umTm_x = umTm_x.rechunk(chunks={0: tile_size/10, 1: tile_size/10, 2: nt})
    vmTm_y = vmTm_y.rechunk(chunks={0: tile_size/10, 1: tile_size/10, 2: nt})
    
    print('p shape after chunking/10 = ',T.shape)
        
    # Function that detrends, windows, and takes fft
    def fft_block(var):
        var = detrend_func(var,'time')
        var = window_func(var,'time')
        varhat = (1./var.shape[2])*np.fft.rfft(var, axis=2)
        return varhat
    
    # Execute fft function above
    That = T.map_blocks(fft_block)
    umTm_xhat = umTm_x.map_blocks(fft_block)
    vmTm_yhat = vmTm_y.map_blocks(fft_block)
    
    That = That.rechunk(chunks={0: tile_size, 1:tile_size, 2: 365})
    umTm_xhat = umTm_xhat.rechunk(chunks={0: tile_size, 1:tile_size, 2: 365})
    vmTm_yhat = vmTm_yhat.rechunk(chunks={0: tile_size, 1:tile_size, 2: 365})

    # Multiply by constants
    TumTm_x = ((That.conj()*umTm_xhat).real)
    TvmTm_y = ((That.conj()*vmTm_yhat).real)
    
    print('final shape = ',TumTm_x.shape)

    # Sum over x- and y-axes
    TumTm_x = da.sum(TumTm_x, axis=(0,1))
    TvmTm_y = da.sum(TvmTm_y, axis=(0,1))
    
    #H_adv = H_adv.compute()
    TumTm_x = TumTm_x.compute()
    TvmTm_y = TvmTm_y.compute()
    
    # Wrap as DataArray
    n = len(time)
    d = time[1] - time[0]
    freq = np.fft.rfftfreq(n, d)
    
    TumTm_x = xr.DataArray(TumTm_x, 
                      dims=['freq'], 
                      coords={'freq': freq,
                              'xt': xt.mean(),
                              'yt': yt.mean()})
    
    TvmTm_y = xr.DataArray(TvmTm_y, 
                      dims=['freq'], 
                      coords={'freq': freq,
                              'xt': xt.mean(),
                              'yt': yt.mean()})
    
    return TumTm_x,TvmTm_y


# ### Vertical temperature advection
# 
# 
# $$
# \frac{1}{H_m} \widehat{T^o}^* \widehat{ w_{ek} T_m}
# $$
# 

# In[6]:


def calc_vert_adv(dsx, tile_index, z=0):
    
    # Select variables
    T = dsx.sst
    wekt = dsx.wekt
    
    # Work with DaskArrays
    ix, iy = tile_index
    T = T.isel(yt=slice(max(iy*tile_size,0), (iy+1)*tile_size),
                 xt=slice(max(ix*tile_size,0), (ix+1)*tile_size))
    wekt = wekt.isel(yt=slice(max(iy*tile_size,0), (iy+1)*tile_size),
                 xt=slice(max(ix*tile_size,0), (ix+1)*tile_size))

    xt = T.xt.values
    yt = T.yt.values
    time = T.time.values
    
    T = T.data
    wekt = wekt.data
    
    print('original shapes of wekt and T = ',wekt.shape,T.shape)
    
    ny, nx, nt = T.shape
    T = T.rechunk(chunks={0: ny, 1: nx})
    wekt = wekt.rechunk(chunks={0: ny, 1: nx})

    ## Mask boundaries, if desired (mask = number of cells to mask on each boundary)
    #if ix == 0 and mask:
    #    T = T[:,slice(mask,None),:]
    #    wekt = wekt[:,slice(mask,None),:]
    #if iy == 0 and mask:
    #    T = T[slice(mask,None),:,:]
    #    wekt = wekt[slice(mask,None),:,:]
    #if ix == 9 and mask:
    #    T = T[:,slice(0,-mask),:] 
    #    wekt = wekt[:,slice(0,-mask),:]
    #if iy == 9 and mask:
    #    T = T[slice(0,-mask),:,:]
    #    wekt = wekt[slice(0,-mask),:,:]
        
    print('wekt shape = ',wekt.shape,T.shape)
    wekT = wekt * T
    
    
    # Take fft (after detrend and window)
    def fft_block(var): # var is numpy array
        var = detrend_func(var,'time')
        var = window_func(var,'time')
        varhat = (1./var.shape[2])*np.fft.rfft(var, axis=2)
        return varhat
    
    # Use smaller spatial chunks for fft
    T = T.rechunk(chunks={0: tile_size/10, 1: tile_size/10, 2: nt})
    wekT = wekT.rechunk(chunks={0: tile_size/10, 
                                  1: tile_size/10, 
                                  2: nt})
    
    # Execute fft function above
    That = T.map_blocks(fft_block)
    wekThat = wekT.map_blocks(fft_block)
    
    #pdiffhat = da.fft.rfft(pdiff, axis=2)
    That = That.rechunk( chunks={0: tile_size, 1:tile_size, 2: 365})
    wekThat = wekThat.rechunk(chunks={0: tile_size, 
                                        1: tile_size, 
                                        2: 365})
    # Multiply by constants
    vert_adv = (1./Hm) * (That.conj()*wekThat).real

    # Sum over x- and y-axes
    vert_adv = da.sum(vert_adv, axis=(0,1))
    
    vert_adv = vert_adv.compute()
    
    # wrap as DataArray
    n = len(time)
    d = time[1] - time[0]
    freq = np.fft.rfftfreq(n, d)
    
    vert_adv = xr.DataArray(vert_adv,
                      dims=['freq'], 
                      coords={'freq': freq,
                              'xt': xt.mean(),
                              'yt': yt.mean()})
    return vert_adv


# ## All terms of the form:
# 
# $$
# {}^o \widehat{T}^* {}^a \widehat{T}
# $$
# 

# In[10]:


def calc_ToTa(dsx1, dsx2, tile_index):
    # Select variables
    To = dsx1.sst
    Ta = dsx2.Tma
    print('Ta shape = ',Ta.shape)
    print('To shape = ',To.shape)
    
    # Work with DaskArrays
    ix, iy = tile_index
    print('ix and iy = ',ix,iy)
    To = To.isel(yt=slice(iy*tile_size, (iy+1)*tile_size),
                 xt=slice(ix*tile_size, (ix+1)*tile_size))
    Ta = Ta.isel(yt=slice(iy*tile_size, (iy+1)*tile_size),
                 xt=slice(ix*tile_size, (ix+1)*tile_size))
    
    print('To and Ta shape after isel = ',To.shape,Ta.shape)
    
    xt = To.xt.values
    yt = To.yt.values
    time = To.time.values
    
    To = To.data
    Ta = Ta.data
    
    ny, nx, nt = To.shape
    
    #Ta = bilint()
    
    print('Original shape of To and Ta = ',To.shape,Ta.shape)
    
    ## Mask boundaries, if desired (mask = number of cells to mask on each boundary)
    #if ix == 0 and mask:
    #    To = To[:,slice(mask,None),:]
    #    Ta = Ta[:,slice(mask,None),:]
    #if iy == 0 and mask:
    #    To = To[slice(mask,None),:,:]
    #    Ta = Ta[slice(mask,None),:,:]
    #if ix == 9 and mask:
    #    To = To[:,slice(0,-mask),:]
    #    Ta = Ta[:,slice(0,-mask),:]
    #if iy == 9 and mask:
    #    To = To[slice(0,-mask),:,:]
    #    Ta = Ta[slice(0,-mask),:,:]
        
    print('Aftermasking = ',To.shape, Ta.shape)
 
    # Take fft (after detrend and window)
    def fft_block(var): # var is numpy array
        var = detrend_func(var,'time')
        var = window_func(var,'time')
        varhat = (1.0/var.shape[2])*np.fft.rfft(var, axis=2)
        return varhat
    
    # Resize for faster FFT
    To = To.rechunk(chunks={0: tile_size/10, 1: tile_size/10, 2: nt})
    Ta = Ta.rechunk(chunks={0: tile_size/10, 1: tile_size/10, 2: nt})

    # Execute fft function above
    Tohat = To.map_blocks(fft_block)
    Tahat = Ta.map_blocks(fft_block)
    
    # Resize back to original tile size (100)
    Tohat = Tohat.rechunk(chunks={0: tile_size, 
                                  1: tile_size, 
                                  2: 365})
    Tahat = Tahat.rechunk(chunks={0: tile_size, 
                                  1: tile_size, 
                                  2: 365})
    
    print('Tohat and Tahat shapes = ',Tohat.shape,Tahat.shape)
    
    # Multiply by constants
    ToTa = (Tohat.conj()*Tahat).real

    # Sum over x- and y-axes
    ToTa = da.sum(ToTa, axis=(0,1))
    
    ToTa = ToTa.compute()
    
    # wrap as DataArray
    n = len(time)
    d = time[1] - time[0]
    freq = np.fft.rfftfreq(n, d)
    
    ToTa = xr.DataArray(ToTa,
                      dims=['freq'], 
                      coords={'freq': freq,
                              'xt': xt.mean(),
                              'yt': yt.mean()})
    
    return ToTa


# ## All terms of the form:
# 
# $$
# {}^o \widehat{T}^* {}^o \widehat{T}
# $$
# 

# In[6]:


def calc_To2(dsx1, tile_index):
    # Select variables
    To = dsx1.sst
    
    # Work with DaskArrays
    ix, iy = tile_index
    To = To.isel(yt=slice(iy*tile_size, (iy+1)*tile_size),
                 xt=slice(ix*tile_size, (ix+1)*tile_size))
    
    print('To shape after isel = ',To.shape)
    
    xt = To.xt.values
    yt = To.yt.values
    time = To.time.values
    
    To = To.data
    
    ny, nx, nt = To.shape
    
    print('Original shape = ',To.shape)
    
    ## Mask boundaries, if desired (mask = number of cells to mask on each boundary)
    #if ix == 0 and mask:
    #    To = To[:,slice(mask,None),:]
    #if iy == 0 and mask:
    #    To = To[slice(mask,None),:,:]
    #if ix == 9 and mask:
    #    To = To[:,slice(0,-mask),:]
    #if iy == 9 and mask:
    #    To = To[slice(0,-mask),:,:]
        
    print('Aftermasking = ',To.shape)
 
    # Take fft (after detrend and window)
    def fft_block(var): # var is numpy array
        var = detrend_func(var,'time')
        var = window_func(var,'time')
        varhat = (1.0/var.shape[2])*np.fft.rfft(var, axis=2)
        return varhat
    
    # Resize for faster FFT
    To = To.rechunk(chunks={0: tile_size/10, 1: tile_size/10, 2: nt})

    # Execute fft function above
    Tohat = To.map_blocks(fft_block)
    
    # Resize back to original tile size (100)
    Tohat = Tohat.rechunk(chunks={0: tile_size, 
                                  1: tile_size, 
                                  2: 365})
    
    # Multiply by constants
    To2 = (Tohat.conj()*Tohat).real

    # Sum over x- and y-axes
    To2 = da.sum(To2, axis=(0,1))
    
    To2 = To2.compute()
    
    # wrap as DataArray
    n = len(time)
    d = time[1] - time[0]
    freq = np.fft.rfftfreq(n, d)
    
    To2 = xr.DataArray(To2,
                      dims=['freq'], 
                      coords={'freq': freq,
                              'xt': xt.mean(),
                              'yt': yt.mean()})
    
    return To2


# # Calculate the Fme term
# 
# $$
# \widehat{T^o}^* \widehat{ w_{ek} (T_m - T_1)}
# $$

# In[7]:


def calc_Fme(dsx, tile_index, z=0):
    
    # Select variables
    T = dsx.sst
    wekt = dsx.wekt
    
    # Work with DaskArrays
    ix, iy = tile_index
    T = T.isel(yt=slice(max(iy*tile_size,0), (iy+1)*tile_size),
                 xt=slice(max(ix*tile_size,0), (ix+1)*tile_size))
    wekt = wekt.isel(yt=slice(max(iy*tile_size,0), (iy+1)*tile_size),
                 xt=slice(max(ix*tile_size,0), (ix+1)*tile_size))

    xt = T.xt.values
    yt = T.yt.values
    time = T.time.values
    
    T = T.data
    wekt = wekt.data
    
    print('original shapes of wekt and T = ',wekt.shape,T.shape)
    
    ny, nx, nt = T.shape
    T = T.rechunk(chunks={0: ny, 1: nx})
    wekt = wekt.rechunk(chunks={0: ny, 1: nx})

    ## Mask boundaries, if desired (mask = number of cells to mask on each boundary)
    #if ix == 0 and mask:
    #    T = T[:,slice(mask,None),:]
    #    wekt = wekt[:,slice(mask,None),:]
    #if iy == 0 and mask:
    #    T = T[slice(mask,None),:,:]
    #    wekt = wekt[slice(mask,None),:,:]
    #if ix == 9 and mask:
    #    T = T[:,slice(0,-mask),:] 
    #    wekt = wekt[:,slice(0,-mask),:]
    #if iy == 9 and mask:
    #    T = T[slice(0,-mask),:,:]
    #    wekt = wekt[slice(0,-mask),:,:]
        
    print('wekt shape = ',wekt.shape,T.shape)
    wekT = wekt * (T - toc1)
    
    
    # Take fft (after detrend and window)
    def fft_block(var): # var is numpy array
        var = detrend_func(var,'time')
        var = window_func(var,'time')
        varhat = (1./var.shape[2])*np.fft.rfft(var, axis=2)
        return varhat
    
    # Use smaller spatial chunks for fft
    T = T.rechunk(chunks={0: tile_size/10, 1: tile_size/10, 2: nt})
    wekT = wekT.rechunk(chunks={0: tile_size/10, 
                                  1: tile_size/10, 
                                  2: nt})
    
    # Execute fft function above
    That = T.map_blocks(fft_block)
    wekThat = wekT.map_blocks(fft_block)
    
    #pdiffhat = da.fft.rfft(pdiff, axis=2)
    That = That.rechunk( chunks={0: tile_size, 1:tile_size, 2: 365})
    wekThat = wekThat.rechunk(chunks={0: tile_size, 
                                        1: tile_size, 
                                        2: 365})
    # Multiply by constants
    Fme= (That.conj()*wekThat).real

    # Sum over x- and y-axes
    Fme = da.sum(Fme, axis=(0,1))
    
    Fme = Fme.compute()
    
    # wrap as DataArray
    n = len(time)
    d = time[1] - time[0]
    freq = np.fft.rfftfreq(n, d)
    
    Fme = xr.DataArray(Fme,
                      dims=['freq'], 
                      coords={'freq': freq,
                              'xt': xt.mean(),
                              'yt': yt.mean()})
    return Fme


# # Calculate sensible and latent heat 
# 
# $$
# \widehat{T^o}^* \widehat{(T^o - T^a)}
# $$

# In[10]:


def calc_slh(dsx1, dsx2, tile_index):
    # Select variables
    To = dsx1.sst
    Ta = dsx2.Tma
    print('Ta shape = ',Ta.shape)
    print('To shape = ',To.shape)
    
    # Work with DaskArrays
    ix, iy = tile_index
    print('ix and iy = ',ix,iy)
    To = To.isel(yt=slice(iy*tile_size, (iy+1)*tile_size),
                 xt=slice(ix*tile_size, (ix+1)*tile_size))
    Ta = Ta.isel(yt=slice(iy*tile_size, (iy+1)*tile_size),
                 xt=slice(ix*tile_size, (ix+1)*tile_size))
    
    print('To and Ta shape after isel = ',To.shape,Ta.shape)
    
    xt = To.xt.values
    yt = To.yt.values
    time = To.time.values
    
    To = To.data
    Ta = Ta.data
    
    ny, nx, nt = To.shape
    
    To_minus_Ta = To - Ta
    
    #Ta = bilint()
    
    print('Original shape of To and Ta = ',To.shape,Ta.shape)
    
    ## Mask boundaries, if desired (mask = number of cells to mask on each boundary)
    #if ix == 0 and mask:
    #    To = To[:,slice(mask,None),:]
    #    To_minus_Ta = To_minus_Ta[:,slice(mask,None),:]
    #if iy == 0 and mask:
    #    To = To[slice(mask,None),:,:]
    #    To_minus_Ta = To_minus_Ta[slice(mask,None),:,:]
    #if ix == 9 and mask:
    #    To = To[:,slice(0,-mask),:]
    #    To_minus_Ta = To_minus_Ta[:,slice(0,-mask),:]
    #if iy == 9 and mask:
    #    To = To[slice(0,-mask),:,:]
    #    To_minus_Ta = To_minus_Ta[slice(0,-mask),:,:]
        
    print('Aftermasking = ',To.shape, Ta.shape)
 
    # Take fft (after detrend and window)
    def fft_block(var): # var is numpy array
        var = detrend_func(var,'time')
        var = window_func(var,'time')
        varhat = (1.0/var.shape[2])*np.fft.rfft(var, axis=2)
        return varhat
    
    # Resize for faster FFT
    To = To.rechunk(chunks={0: tile_size/10, 1: tile_size/10, 2: nt})
    To_minus_Ta = To_minus_Ta.rechunk(chunks={0: tile_size/10, 1: tile_size/10, 2: nt})

    # Execute fft function above
    Tohat = To.map_blocks(fft_block)
    Tahat = To_minus_Ta.map_blocks(fft_block)
    
    # Resize back to original tile size (100)
    Tohat = Tohat.rechunk(chunks={0: tile_size, 
                                  1: tile_size, 
                                  2: 365})
    Tahat = Tahat.rechunk(chunks={0: tile_size, 
                                  1: tile_size, 
                                  2: 365})
    
    print('Tohat and Tahat shapes = ',Tohat.shape,Tahat.shape)
    
    # Multiply by constants
    slh = (Tohat.conj()*Tahat).real

    # Sum over x- and y-axes
    slh = da.sum(slh, axis=(0,1))
    
    slh = slh.compute()
    
    # wrap as DataArray
    n = len(time)
    d = time[1] - time[0]
    freq = np.fft.rfftfreq(n, d)
    
    slh = xr.DataArray(slh,
                      dims=['freq'], 
                      coords={'freq': freq,
                              'xt': xt.mean(),
                              'yt': yt.mean()})
    
    return slh


# # Calculate grad squared term
# 
# $$
# \widehat{T^o}^* \widehat{\nabla ^2 T^o}
# $$

# In[8]:


def calc_Tdel2T(dsx1, tile_index):
    # Select variables
    To = dsx1.sst
    
    # Work with DaskArrays
    ix, iy = tile_index
    To = To.isel(yt=slice(max(iy*tile_size-1,0), (iy+1)*tile_size+1),
               xt=slice(max(ix*tile_size-1,0), (ix+1)*tile_size+1))
    
    print('To shape after isel = ',To.shape)
    
    xt = To.xt.values
    yt = To.yt.values
    time = To.time.values
    
    To = To.data
    
    ny, nx, nt = To.shape
    
    print('Original shape = ',To.shape)
    
    # Take a second derivative
    def Derivative(var, axis):
        var_right = np.roll(var, 1, axis=axis)
        var_left = np.roll(var, -1, axis=axis)
        dvar = var_left + var_right - 2*var
        return dvar
    
    Derivative_x = lambda p: Derivative(p, 1)
    Derivative_y = lambda p: Derivative(p, 0)
    
    # Evaluate the derivative function
    del2_x = (1./(dx**2)) * To.map_blocks(Derivative_x)
    del2_y = (1./(dx**2)) * To.map_blocks(Derivative_y)
    
    print('del2x shape = ',del2_x.shape)
    print('del2y shape = ',del2_y.shape)
    print('dx = ',dx)
    
    del2 = del2_x + del2_y
    print('del2.shape = ',del2.shape)
    
    # Take off added cells
    To = To[2:-2,2:-2,:]
    del2 = del2[2:-2,2:-2,:]
    
    ## Define mask to be 1 here since I slice off ends above
    #if iy == 0:
    #    To = To[1:, :, :]
    #    del2 = del2[1:, :, :]
    #if iy == 9:
    #    To = To[:-1, :, :]
    #    del2 = del2[:-1, :, :]
    #if ix == 0:
    #    To = To[:, 1:, :]
    #    del2 = del2[:, 1:, :]
    #if ix == 9:
    #    To = To[:, :-1, :]
    #    del2 = del2[:, :-1, :]  
    print('Aftermasking = ',To.shape,del2.shape)
 
    # Take fft (after detrend and window)
    def fft_block(var): # var is numpy array
        var = detrend_func(var,'time')
        var = window_func(var,'time')
        varhat = (1.0/var.shape[2])*np.fft.rfft(var, axis=2)
        return varhat
    
    # Resize for faster FFT
    To = To.rechunk(chunks={0: tile_size/10, 1: tile_size/10, 2: nt})
    del2 = del2.rechunk(chunks={0: tile_size/10, 1: tile_size/10, 2: nt})

    # Execute fft function above
    Tohat = To.map_blocks(fft_block)
    del2hat = del2.map_blocks(fft_block)
    
    # Resize back to original tile size (100)
    Tohat = Tohat.rechunk(chunks={0: tile_size, 
                                  1: tile_size, 
                                  2: 365})
    del2hat = del2hat.rechunk(chunks={0: tile_size, 
                                  1: tile_size, 
                                  2: 365})
    
    # Multiply by constants
    Tdel2T = (Tohat.conj()*del2hat).real

    # Sum over x- and y-axes
    Tdel2T = da.sum(Tdel2T, axis=(0,1))
    
    Tdel2T = Tdel2T.compute()
    
    # wrap as DataArray
    n = len(time)
    d = time[1] - time[0]
    freq = np.fft.rfftfreq(n, d)
    
    Tdel2T = xr.DataArray(Tdel2T,
                      dims=['freq'], 
                      coords={'freq': freq,
                              'xt': xt.mean(),
                              'yt': yt.mean()})
    
    return Tdel2T


# # Calculate grad fourth term
# 
# $$
# \widehat{T^o}^* \widehat{\nabla ^4 T^o}
# $$

# In[9]:


def calc_Tdel4T(dsx1, tile_index):
    # Select variables
    To = dsx1.sst
    
    # Work with DaskArrays
    ix, iy = tile_index
    To = To.isel(yt=slice(max(iy*tile_size-2,0), (iy+1)*tile_size+2),
               xt=slice(max(ix*tile_size-2,0), (ix+1)*tile_size+2))
    
    print('To shape after isel = ',To.shape)
    
    xt = To.xt.values
    yt = To.yt.values
    time = To.time.values
    
    To = To.data
    
    ny, nx, nt = To.shape
    
    print('Original shape = ',To.shape)
    
    # Take a fourth derivative
    def Derivative(var, axis):
        var_right = np.roll(var, 1, axis=axis)
        var_2right = np.roll(var, 2, axis=axis)
        var_left = np.roll(var, -1, axis=axis)
        var_2left = np.roll(var, -2, axis=axis)
        dvar = var_2left - 4*var_left + 6*var - 4*var_right + var_2right
        return dvar
        
    Derivative_x = lambda p: Derivative(p, 1)
    Derivative_y = lambda p: Derivative(p, 0)
    
    # Evaluate the derivative function
    del4_x = (1./(dx**4)) * To.map_blocks(Derivative_x)
    del4_y = (1./(dx**4)) * To.map_blocks(Derivative_y)
    
    print('dpdx shape = ',del4_x.shape)
    print('tauy_avg shape = ',del4_y.shape)
    
    del4 = del4_x + del4_y
    
    print('del4.shape = ',del4.shape)
    
    # Take off added cells
    To = To[4:-4,4:-4,:]
    del4 = del4[4:-4,4:-4,:]
    print('Aftermasking = ',To.shape,del4.shape)
 
    # Take fft (after detrend and window)
    def fft_block(var): # var is numpy array
        var = detrend_func(var,'time')
        var = window_func(var,'time')
        varhat = (1.0/var.shape[2])*np.fft.rfft(var, axis=2)
        return varhat
    
    # Resize for faster FFT
    To = To.rechunk(chunks={0: tile_size/10, 1: tile_size/10, 2: nt})
    del4 = del4.rechunk(chunks={0: tile_size/10, 1: tile_size/10, 2: nt})

    # Execute fft function above
    Tohat = To.map_blocks(fft_block)
    del4hat = del4.map_blocks(fft_block)
    
    # Resize back to original tile size (100)
    Tohat = Tohat.rechunk(chunks={0: tile_size, 
                                  1: tile_size, 
                                  2: 365})
    del4hat = del4hat.rechunk(chunks={0: tile_size, 
                                  1: tile_size, 
                                  2: 365})
    
    # Multiply by constants
    Tdel4T = (Tohat.conj()*del4hat).real

    # Sum over x- and y-axes
    Tdel4T = da.sum(Tdel4T, axis=(0,1))
    
    Tdel4T = Tdel4T.compute()
    
    # wrap as DataArray
    n = len(time)
    d = time[1] - time[0]
    freq = np.fft.rfftfreq(n, d)
    
    Tdel4T = xr.DataArray(Tdel4T,
                      dims=['freq'], 
                      coords={'freq': freq,
                              'xt': xt.mean(),
                              'yt': yt.mean()})
    
    return Tdel4T


    
