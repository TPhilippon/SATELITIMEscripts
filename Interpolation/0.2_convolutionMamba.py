# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 14:55:40 2016

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
from astropy.convolution import convolve, Gaussian2DKernel
from matplotlib.colors import Colormap
from mamba import *
import mambaDisplay.extra


#==============================================================================
#                             DEFINITIONS
#==============================================================================

homepath = os.environ['HOMEPATH']  # Windows = os.environ['HOMEPATH'] ;;; Linux = os.environ['HOME']
path = homepath+'//SATELITIME//data//ZR//'  # Windows = '//SATELITIME//data//ZR//' ;;; Linux = '/SATELITIME/data/ZR/'
path2 = homepath+'//SATELITIME//data//convolve//'
outpath = homepath+'//SATELITIME//data//convolve//'  # Windows = //SATELITIME//data//Output//' ;;; Linux = '/SATELITIME/data/Output/'

# **Colormap Chl de ref**

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
# ****

#path = '/Users/terencephilippon/Desktop/Python/Input/'
#outpath = '/Users/terencephilippon/Desktop/Python/Output/'
print 'starting...'
print path


# Data we want to read
data = glob.glob(path+'*.npy')
data.sort()
print data

# Def kernels
gauss = Gaussian2DKernel(stddev=1)
#gauss_fft = Gaussian2DKernel(stddev=1)

#==============================================================================
#                               LOOP
#==============================================================================

for myfile in data:
    print 'reading data...'
    print myfile
    zr = np.load(myfile)
    
#==============================================================================
#                               CONVOLVE    
#==============================================================================
    
    zr_conv = convolve(zr,gauss)
#    zr_convfft = convolve_fft(zr,gauss_fft)
    
#    fig1 = plt.gcf()
#    fig, (ax1, ax2) = plt.subplots(1,2)
    fig, (ax1) = plt.subplots(1,1)
#    plt.imshow(zr_conv, norm=norm_chl, origin='upper', cmap=new_map_chl,)
#    ax1.imshow(zr, norm=norm_chl, origin='upper', cmap=new_map_chl,)
    ax1.imshow(zr_conv, norm=norm_chl, origin='upper', cmap=new_map_gray_chl,) 
#    ax3.imshow(zr_convfft, norm=norm_chl, origin='upper', cmap=new_map_chl,) 
    plt.show()
    fig.savefig(outpath+myfile[-46:-4]+'_convolve'+'.npy', dpi=200, bbox_inches='tight')
    plt.close()

#==============================================================================
#                                 MAMBA 
#==============================================================================
    
data2 = glob.glob(path2+'*Gray.png')
data2.sort()
print data2
    
for myfile in data2:
    
    im = imageMb(myfile)
    imSeg = imageMb(im, 32)
    print(mambaDisplay.extra.interactiveSegment(im, imSeg))
    
    
#    fig, (ax1, ax2) = plt.subplots(1,2)
##    plt.imshow(zr_conv, norm=norm_chl, origin='upper', cmap=new_map_chl,)
#    ax1.imshow(zr, norm=norm_chl, origin='upper', cmap=new_map_chl,)
#    ax2.imshow(zr_conv, norm=norm_chl, origin='upper', cmap=new_map_chl,) 
##    ax3.imshow(zr_convfft, norm=norm_chl, origin='upper', cmap=new_map_chl,) 
#    plt.show()
#    fig.savefig(outpath+myfile[-46:-4]+'_convolve'+'.png', dpi=200, bbox_inches='tight')
#    plt.close()

















