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
from scipy.interpolate import LinearNDInterpolator, griddata
from matplotlib.colors import Colormap
from astropy.convolution import convolve, Gaussian2DKernel

#==============================================================================
# #                             Definitions 
#==============================================================================

#varhomepath = 1   # Windows = 0 ;;; Linux/MacOS = 1 
varInterpolation = 1   # Nearest = 0 ;;; Linear = 1

if os.name == 'posix':
    varhomepath = 1
else: varhomepath = 0

homepath = os.environ['HOME']  # Windows = os.environ['HOMEPATH'] ;;; Linux = os.environ['HOME']
path = homepath+'/SATELITIME/data/ZR/'
outpath = homepath+'/SATELITIME/data/contours/interp/'

gauss = Gaussian2DKernel(stddev=1)

#path = '/Users/terencephilippon/Desktop/Python/Input/'
#outpath = '/Users/terencephilippon/Desktop/Python/Output/'
print 'starting...'
print path

# Landmask to find land releted NaN to exclude from interpolation.
landmask = plt.imread('/Users/terencephilippon/Desktop/Python/'+'landmaskZRbw.png') # préfixe
# Landmask de forme 350,500,4 ; la matrice est dans le 4ième indice (indice 3 donc pour 0,1,2,3)
lignes, colonnes, indice = landmask.shape
landmask3 = landmask[:lignes,:colonnes,3]  #!!! [0:350,0:500,3] Code avec long, larg, rgb = landmask.shape
#sLandNAN = np.select(landmask3>0.1,landmask3)
landXY = np.where(landmask3==1)
landXY = np.asarray(landXY).T
#x = landXY[0]
#y = landXY[1]

landmaskNAN = landmask3
landmaskNAN[landmaskNAN>0.5] = np.nan
#landmaskNAN[landmaskNAN<0.5] = 0

landNAN = np.empty(shape=(landXY.shape[0],1)) ###
landNAN[:] = np.nan

#x = x.reshape(landXY.shape[1],1)
#y = y.reshape(landXY.shape[1],1)
#landXY = np.hstack((x,y))

landmask3[landmask3>0.5] = 999
landmask3[landmask3<0.5] = 0

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
    zrSOURCE = np.load(myfile)
    
    zr = convolve(zrSOURCE,gauss)
    
    #Obtenir ZR en 3 colonnes :
#    zrCONV = convolve(zr,gauss)
    #1. Coordonnées de zr.
    coords = np.argwhere(zr)
    #2. Coordonnées en colonne.
#    xcoords = coords[:,0]
#    ycoords = coords[:,1]
    #3. Valeurs de zr en ligne.
    zrxyt = zr[[coords[:,0]],[coords[:,1]]].T #
    #Transpose de ZR pour passer en colonne.
#    zrxyt=zrxy.T
    #4. Ajouter la colonne ZR aux colonnes des coordonnes (3c total).
    zr3 = np.hstack((coords,zrxyt))
    #Extract zr3 values only.
    zrON = zr3[~np.isnan(zr3[:,2])]
    
#==============================================================================
#           #*****Exclusion des NaN correspondants au continent******
#==============================================================================
    
    zr[np.isnan(zr)] = 0
    
    zrNEW = zr+landmask3   # land >= 999 ;;; NaN to interpolate == 0
    
    #
    zrland = zrNEW[zrNEW>=999]
    zrlandxy = np.argwhere(zrNEW>=999)
    zrland = zrland.reshape(zrlandxy.shape[0],1)
    zrland3 = np.hstack((zrlandxy,zrland))    
    zrland3[:,2] = np.nan    # zrland3 = coords and NaN for landmask
    
    zrNEW[zrNEW==0] = np.nan # Retour NaN sur pixels à interpoler
    
    zrNAN = zrNEW[np.isnan(zrNEW)]
    zrNANxy = np.argwhere(np.isnan(zrNEW))
    zrNAN = zrNAN.reshape(zrNANxy.shape[0],1)
    zrNAN3col = np.hstack((zrNANxy,zrNAN))

        
    
#==============================================================================
#                         ***** INTERPOLATION ******
#==============================================================================
   
    grid_z0 = griddata((zrON[:,0], zrON[:,1]), zrON[:,2], zrNANxy, method='linear')   
   
    matrix = np.zeros(zr.shape)
    matrix[zrNANxy[:,0],zrNANxy[:,1]]= grid_z0
    matrix[zrON[:,0].astype(int),zrON[:,1].astype(int)]=zrON[:,2]
   
    matrix = matrix+landmaskNAN
#    Create interpolator -* Choose interpolator *-
#    # $$$
#    interp0 = [NearestNDInterpolator(zrON[:,0:2],zrON[:,2]), 
#               LinearNDInterpolator(zrON[:,0:2],zrON[:,2])][varInterpolation]
#             
#    #Apply interpolator to coordinates.
#    interp=interp0(zrNAN3col[:,0:2])
##    interp=interp0((zrNAN[:,0],zrNAN[:,1]),zrNAN[:,2])
##    result0=interp0(np.ravel(xx), np.ravel(yy)).reshape( xx.shape )
#    
#    zrNEW2=np.hstack((zrNAN3col[:,0:2],interp[:,None])) # Je colle la 3e colonne des données interpolées.
#    zrINT=np.vstack((zrON,zrNEW2)) # All originale + Interpolated : format np.array (x,y,z)
#    zrINT=np.vstack((zrINT,zrland3))
#    # Converntir en matrice-image contraire de aregwhere
#    X = zrINT[:,0].astype(int)
#    Y = zrINT[:,1].astype(int)
#    Z = zrINT[:,2]
#
#    X=np.array(X).tolist()
#    Y=np.array(Y).tolist()
#    matrix = np.zeros(zr.shape)
#    matrix[X,Y] = Z
#     $$$

    # Afficher / enregistrer l'image
    fig1 = plt.gcf()
    plt.imshow(matrix, norm=norm_chl, origin='upper', cmap=new_map_chl,)
#    plt.imshow(landmask,)
    plt.show()
#    fig1.savefig(outpath+myfile[-46:-4]+'interpConvolve'+'.png')
    plt.close()

    print myfile+' -----> done'

print '>>> end'


    
    
