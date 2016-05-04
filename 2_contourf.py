# -*- coding: utf-8 -*-
"""
Created on Mon May  2 18:09:45 2016

@author: terencephilippon
"""

# Loading image files ; create isolines ; plot

import os, sys
import glob
import numpy as np
from pylab import mpl as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
from os.path import basename
from mamba import *
from scipy import ndimage

#==============================================================================
# #                             Definitions 
#==============================================================================

#varhomepath = 1   # Windows = 0 ;;; Linux/MacOS = 1 
varInterpolation = 1   # Nearest = 0 ;;; Linear = 1

if os.name == 'posix':
    homepath = os.environ['HOME']
else: homepath = os.environ['HOMEPATH']

#homepath = os.environ['HOME']  # Windows = os.environ['HOMEPATH'] ;;; Linux = os.environ['HOME']
if os.name == 'nt':
    path = homepath+'\\SATELITIME\\data\\contours\\interp_npy\\'
    outpath = homepath+'\\SATELITIME\\data\\contours\\iso_npy\\'

else : 
    path = homepath+'/SATELITIME/data/contours/interp_npy/'
    outpath = homepath+'/SATELITIME/data/contours/iso_npy/'

    
#path = homepath+'/SATELITIME/data/ZR/'
#outpath = homepath+'/SATELITIME/data/contours/interp_png/'
#outpathNPY = homepath+'/SATELITIME/data/contours/interp_npy/'

#path = '/Users/terencephilippon/Desktop/Python/Input/'
#outpath = '/Users/terencephilippon/Desktop/Python/Output/'
print 'starting...'
print path

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


#==============================================================================
# #                             Starting loop
#==============================================================================


#arr = np.array([[[1,2], [3,5]], [[4,8], [1,9]]])

# Création array vide pour stocker les données. 
#x = np.zeros(10, dtype= [('date', 'S15', 1), ('seuil', 'int8', 6), ('zrSEUIL', 'int8', 1)])

# Rappel : seuilx100 
seuils = np.array([(0.05), (0.10), (0.15), (0.20), (0.30), (0.40), (10)])
#seuils = np.array([(0.20), (0.40)])
#Seuils = seuils*100


i = 0

for myfile in data:
    print 'reading data...'
    print myfile
    r = np.load(myfile)                        # numpy
#    x['date'][i] = basename(myfile[:-4])
#    x['seuil'][i] = Seuils
#    x['zrSEUIL'][i] = r
#    i += 1
    
    zrcontour = plt.contourf(r,seuils, origin='upper')
#    plt.contour(r,seuils, origin='upper')
    A = ndimage.binary_opening(r)
    plt.plot()
    # --- MbImage ---
#    im = imageMb(myfile)
#    negate(im, im)
#    closeHoles(im,im)
#    plt.imshow(im)
#    im.save(outpath+basename(myfile[:-4])+'.png')    
    # --- MbImage ---
    
#    fig1 = plt.gcf()
    plt.imshow(zrcontour, norm=norm_chl, origin='upper', cmap=new_map_chl,)
#    plt.show()
    
print 'end'







