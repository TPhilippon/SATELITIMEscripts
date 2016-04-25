# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os,sys
#from pyhdf.SD import SD, SDC   
from pylab import mpl as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
#from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import time
import glob

from scipy.interpolate import LinearNDInterpolator
#import pyKriging  
#from pyKriging.krige import kriging  
#from pyKriging.samplingplan import samplingplan

# =========================================================================
# =========================== Definition Varaibles ========================
# =========================================================================

varnum = 3
path = '/Users/terencephilippon/Desktop/Python/Input/'
savepath = '/Users/terencephilippon/Desktop/Python/Output/'

# Colormap Chl de ref
norm_chl=mpl.colors.LogNorm(vmin=0.01, vmax=20)
colors = [(0.33,0.33,0.33)] + [(plt.cm.jet(i)) for i in xrange(1,256)]
new_map_chl = mpl.colors.LinearSegmentedColormap.from_list('new_map_chl', colors, N=256) 
colors = [(0.33,0.33,0.33)] + [(plt.cm.gray(i)) for i in xrange(1,256)]
new_map_gray_chl = mpl.colors.LinearSegmentedColormap.from_list('new_map_gray_chl', colors, N=256) 

# Colormap Couleur et grise
norm = mpl.colors.LogNorm(vmin=0.005, vmax=20)
colors = [(0.33,0.33,0.33)] + [(plt.cm.jet(i)) for i in xrange(1,256)]
new_map = mpl.colors.LinearSegmentedColormap.from_list('new_map', colors, N=256) 
colors = [(0.33,0.33,0.33)] + [(plt.cm.gray(i)) for i in xrange(1,256)]
new_map_gray = mpl.colors.LinearSegmentedColormap.from_list('new_map_gray', colors, N=256)

# =========================================================================
# =========================== Importing Data ==============================
# =========================================================================

print 'starting...'
print path

data = glob.glob(path+'*.npy')
data.sort()

for myfile in data:
    print 'reading data...'
    print myfile
    zr = np.load(myfile)
    


# =========================================================================
# =========================== Interpolating ===============================
# =========================================================================
    
    coords = np.argwhere(zr)
    
    mask = np.isnan(zr)
    
    # array of (number of points, 2) containing the x,y coordinates of the valid values only
    xx, yy = np.meshgrid(np.arange(zr.shape[1]), np.arange(zr.shape[0]))
    xym = np.vstack( (np.ravel(xx[mask]), np.ravel(yy[mask])) ).T
    
    # the valid values in the channel,  as 1D arrays (in the same order as their coordinates in xym)
    zr0 = np.ravel( zr[mask] )
    interp0 = LinearNDInterpolator(xym,zr0)
    result0=interp0(np.ravel(xx), np.ravel(yy)).reshape( xx.shape )
    zr0=result0
    
#    zr = np.ravel( zr[mask] )
#    interp0 = LinearNDInterpolator(xym,zr)
#    result0=interp0(np.ravel(xx), np.ravel(yy)).reshape( xx.shape )
#    zr=result0


# =========================================================================
# =========================== Plot @ save =================================
# =========================================================================


    #plt.imshow(zr,norm=norm, origin='upper', cmap=new_map_gray_chl)
    im2 = plt.imshow(zr0,norm=norm, origin='upper', cmap=new_map_chl)
    im = plt.imshow(zr,norm=norm, origin='upper', cmap=new_map_chl)
    plt.axis('off')
    pngfig = savepath+myfile[-46:-4]+'_iso.'+'png'
    plt.savefig(pngfig,format='png') #,dpi=72,bbox_inches='tight'
    plt.clf()




plt.close()
print 'end.'
