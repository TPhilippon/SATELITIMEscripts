# -*- coding: utf-8 -*-
"""
Created on Mon May  2 18:09:45 2016

@author: terencephilippon
"""

import os, sys
import glob
import numpy as np
from pylab import mpl as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap


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
    outpath = homepath+'\\SATELITIME\\data\\contours\\zrSEUILS\\'

else : 
    path = homepath+'/SATELITIME/data/contours/interp_npy/'
    outpath = homepath+'/SATELITIME/data/contours/zrSEUILS/'

    
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

data = glob.glob(path+'*.npy')
data.sort()
print data

#==============================================================================
# #                             Starting loop
#==============================================================================


#arr = np.array([[[1,2], [3,5]], [[4,8], [1,9]]])

# Création array vide pour stocker les données. 
x = np.zeros(5, dtype= [('date', 'S14', 500), ('seuil', 'int8', 8), ('zrSEUIL', 'int8', (350,500))])

# Rappel : seuilx100 

for myfile in data:
    print 'reading data...'
    print myfile

    








