# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 10:52:55 2016

@author: terencephilippon
"""

import os,sys
#from pyhdf.SD import SD, SDC   
from pylab import mpl as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
#from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import glob
from PIL import Image
from scipy.interpolate import NearestNDInterpolator
from matplotlib.colors import Colormap

#==============================================================================
# #                             Definitions 
#==============================================================================

path = '/Users/terencephilippon/Desktop/Python/Input/'
outpath = '/Users/terencephilippon/Desktop/Python/Output/'
print 'starting...'
print path

# Landmask to find land releted NaN to exclude from interpolation.
landmask = plt.imread('/Users/terencephilippon/Desktop/Python/'+'landmaskZRbw.png')
# Landmask de forme 350,500,4 ; la matrice est dans le 4ième indice (indice 3 donc pour 0,1,2,3)
landmask3 = landmask[0:350,0:500,3]
#sLandNAN = np.select(landmask3>0.1,landmask3)
landXY = np.where(landmask3==1)
landXY = np.asarray(landXY)
x = landXY[0]
y = landXY[1]
x = x.reshape(40007,1) ###
y = y.reshape(40007,1)  ###
landXY = np.hstack((x,y))

landmask3[landmask3>0.5] = 100
landmask3[landmask3<0.5] = 0
#landXY = landXY.reshape(42002,2)

landNAN = np.empty(shape=(40007,1))  ###
landNAN[:] = np.nan
land = np.hstack((landXY,landNAN))

# Data we want to read and interpolate
data = glob.glob(path+'*.npy')
data.sort()
print data

# Colormap Chl de ref

#           COULEUR
norm_chl=mpl.colors.LogNorm(vmin=0.01, vmax=20)
colors = [(0.33,0.33,0.33)] + [(plt.cm.jet(i)) for i in xrange(1,256)]
new_map_chl = mpl.colors.LinearSegmentedColormap.from_list('new_map_chl', colors, N=256)
new_map_chl._init(); new_map_chl._lut[0,:] = new_map_chl._lut[1,:] # Replace lowest value of colormap (which is gray) with the one before (dark blue)
Colormap.set_under(new_map_chl,color=new_map_chl._lut[0,:]) # Set color for values outside colormap to be the lowest of the colormap (dark blue)
##Colormap.set_over(new_map_chl,color=(0.0, 0.0, 0.517825311942959, 1.0))
## to get rgba from colormap for a specific value : new_map_chl(specificvalue for example ex : 0.2)

#           BLACK AND WHITE
grays = [(0.33,0.33,0.33)] + [(plt.cm.gray(i)) for i in xrange(1,256)]
new_map_gray_chl = mpl.colors.LinearSegmentedColormap.from_list('new_map_gray_chl', grays, N=256)

# Colormap (avec borne grise)
#norm_chl=mpl.colors.LogNorm(vmin=0.01, vmax=20)
#colors = [(0.33,0.33,0.33)] + [(plt.cm.jet(i)) for i in xrange(1,256)]
#new_map_chl = mpl.colors.LinearSegmentedColormap.from_list('new_map_chl', colors, N=256) 
#colors = [(0.33,0.33,0.33)] + [(plt.cm.gray(i)) for i in xrange(1,256)]
#new_map_gray_chl = mpl.colors.LinearSegmentedColormap.from_list('new_map_gray_chl', colors, N=256) 

#==============================================================================
# #                             Starting loop
#==============================================================================

for myfile in data:
    print 'reading data...'
    print myfile
    zr = np.load(myfile)
#-----------------------------------------
    zr[np.isnan(zr)] = 999
     
    y = np.ma.masked_values(zr, 999)
#    y.mask
#    array([False, False, False,  True, False, False,  True], dtype=bool)
#    y.data
#    array([ 1,  2,  3, 99,  5,  6, 99])
#    y
#    masked_array(data = [1 2 3 -- 5 6 --], mask = [False False False  True False False  True], fill_value = 99)
     
    #Maks into integer for visual format and purpose (like printing in GIS)  
    intarray = y.mask.astype(int)
#-----------------------------------------
    
#==============================================================================
    #Alternative method if a mask array already exist
    t = np.ma.array(zr, mask=landmask3)
#==============================================================================













