# -*- coding: utf-8 -*-
"""
Created on Mon May  2 18:09:45 2016

@author: terencephilippon
"""

# Morphology ; interpolation 3d.

import os, sys
import glob
import numpy as np
from pylab import mpl as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
from os.path import basename
#from mamba import *
from scipy import ndimage
from pylab import savefig
from scipy.interpolate import LinearNDInterpolator, griddata
import scipy.ndimage
from astropy.convolution import convolve, Gaussian2DKernel


#==============================================================================
# #                             Definitions 
#==============================================================================

#varhomepath = 1   # Windows = 0 ;;; Linux/MacOS = 1 
#varInterpolation = 1   # Nearest = 0 ;;; Linear = 1

if os.name == 'posix':
    homepath = os.environ['HOME']
else: homepath = os.environ['HOMEPATH']

#homepath = os.environ['HOME']  # Windows = os.environ['HOMEPATH'] ;;; Linux = os.environ['HOME']
if os.name == 'nt':
    path = homepath+'\\SATELITIME\\data\\contours\\interp_npy\\'
    outpathNPY = homepath+'\\SATELITIME\\data\\contours\\iso_npy\\'
    outpathPNG = homepath+'\\SATELITIME\\data\\contours\\iso_png\\'
else : 
    path = homepath+'/SATELITIME/data/contours/interp_npy/'
    outpathNPY = homepath+'/SATELITIME/data/contours/iso_npy/'
    outpathPNG = homepath+'/SATELITIME/data/contours/iso_png/'
    
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


# Création array vide pour stocker les données. 
#matrix = np.zeros(10, dtype= [('date', 'S15', 1), ('seuil', 'int8', 6), ('zrSEUIL', 'int8', 1)])
#matrix = np.zeros(10, dtype= [('seuil', 'int8', 6), ('zrSEUIL', 'int8', 1)])

matrix = np.zeros([91, 7, 350,500])

# Alternate : seuilx100 
seuils = np.array([(0.05), (0.10), (0.15), (0.20), (0.30), (0.40), (5)])

iseuil = 0
ifile = 0

kernel4= np.array([[0, 0, 0, 1, 1, 0, 0, 0],
                   [0, 0, 1, 1, 1, 1, 0, 0],
                   [0, 1, 1, 1, 1, 1, 1, 0],
                   [1, 1, 1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1, 1, 1],
                   [0, 1, 1, 1, 1, 1, 1, 0],
                   [0, 0, 1, 1, 1, 1, 0, 0],
                   [0, 0, 0, 1, 1, 0, 0, 0]])

kernel3= np.array([[0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 1, 1, 1, 0, 0],
                   [0, 1, 1, 1, 1, 1, 0],
                   [0, 1, 1, 1, 1, 1, 0],
                   [0, 1, 1, 1, 1, 1, 0],
                   [0, 0, 1, 1, 1, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0]])
                   
kernel2= np.array([[0, 0, 0, 0, 0],
                   [0, 1, 1, 1, 0],
                   [0, 1, 1, 1, 0],
                   [0, 1, 1, 1, 0],
                   [0, 0, 0, 0, 0]])  
                   
kernel1= np.array([[0, 0, 0, 0],
                   [0, 1, 1, 0],
                   [0, 1, 1, 0],
                   [0, 0, 0, 0]]) 
                   
gauss = Gaussian2DKernel(stddev=1)
                
#==============================================================================
# #                             Starting loop
#==============================================================================
                   
for myfile in data:
    print 'reading data...'
    print myfile
    
    # *************
    # SEUILS
    ZR = np.load(myfile)
    
    for iseuil in range(7):
     
        zr = np.ma.masked_array(ZR,ZR>seuils[iseuil])
        zr.data
        zrs = np.ma.masked_array(ZR,ZR>seuils[iseuil]).mask
        
        
        # *********************************************************************
        # Morpho maths
        
        zrd1 = ndimage.binary_dilation(zrs, structure=kernel3).astype(zr.dtype)
        zre1 = ndimage.binary_erosion(zrd1, structure=kernel3).astype(zr.dtype)
#        zre1 = ndimage.binary_erosion(zrs, structure=kernel3).astype(zr.dtype)
#        zrd1 = ndimage.binary_dilation(zre1, structure=kernel3).astype(zr.dtype)
        
        #Closing holes
        result0= scipy.ndimage.morphology.binary_fill_holes(zre1, structure=kernel3)
        
        #Closing islands and headland (cap)
        result= np.invert(result0)
        result1= scipy.ndimage.morphology.binary_closing(result, structure=kernel3)
        result2= np.invert(result1)
        
#        result= np.invert(scipy.ndimage.morphology.binary_fill_holes(result0, structure=kernel1))

        # *********************************************************************
        # Print seuil & save all seuils (npy).
        matrix[ifile, iseuil] = result2
        print "ifile =", ifile, "  iseuil =", iseuil

        fig1 = plt.gcf()
#        plt.imshow(matrix[ifile,iseuil])
        plt.imshow(result2)
        fig1.savefig(outpathPNG+myfile[-46:-4]+'_iso'+'_seuil'+str(iseuil)+'.png')
        plt.close()
        
        
#        matrix[ifile+1:ifile+10,:,:] = np.nan
        
    ifile+=10

print '---> End morpho'
#==============================================================================
# #                             Interpolation 4D
#==============================================================================
#        

print '---> Interpolation 4d'
iseuil = 0
ifile = 0
t = 80

while ifile <= t:
    for iseuil in range(7):
        
        print 'ifile =', ifile+5, '   iseuil =', iseuil
        
        plt.imshow(matrix[ifile,iseuil,:,:])
        plt.show()
        raw_input('mat1')    
        plt.imshow(matrix[ifile+10,iseuil,:,:])
        plt.show()
        raw_input('mat2')   

        mat = (matrix[ifile,iseuil,:,:] + matrix[ifile+10,iseuil,:,:])

        plt.imshow(mat)
        plt.show()
        raw_input('mat12')   
    

        mat[mat==1], mat[mat==0] = np.nan, 1
        
        plt.imshow(mat)
        plt.show()
        raw_input('matnan')   
                
        
        
        
        
#        Alternative 4d
#        astro = convolve(mat,gauss)        
        
        x=np.indices(mat.shape)[0]
        y=np.indices(mat.shape)[1]
        x=x.flatten()
        y=y.flatten()
        m=mat.flatten()
        #        nn=n.flatten()
        #        d0=np.zeros(m.shape)
        #        d10=np.ones(nn.shape)*10
        #        d5=np.ones(nn.shape)*5 
        
        mat3 = np.vstack((x,y))
        mat3 = np.vstack((mat3,m))
        mat3 = mat3.T
        
        coordsNAN = mat3[np.isnan(mat3[:,2])]
        #        coordsNAN = np.argwhere(np.isnan(mat3))
        #        coords = np.argwhere(~np.isnan(mat3))
        coords = mat3[~np.isnan(mat3[:,2])]
        #        xx=np.vstack((x,x)).flatten()
        #        yy=np.vstack((y,y)).flatten()
        #        mn=np.vstack((mm,nn)).flatten()
        #        dd=np.vstack((d0,d10)).flatten()
        
        
        interp = griddata((coords[:,0],coords[:,1]), coords[:,2], coordsNAN[:,0:2], method='cubic')
        
        mat2 = np.zeros(mat.shape)
        mat2[coordsNAN[:,0].astype(int),coordsNAN[:,1].astype(int)]= interp

        plt.imshow(mat2)
        plt.show()
        raw_input("interp")

        mat2[coords[:,0].astype(int),coords[:,1].astype(int)]=coords[:,2]
        
        plt.imshow(mat2)
        plt.show()
        raw_input("interp2")
        
        matrix[ifile+5, iseuil] = mat2    
        
        fig2 = plt.gcf()
#        plt.imshow(matrix[ifile,iseuil])
        plt.imshow(mat2)
        fig2.savefig(outpathPNG+myfile[-46:-4]+'_iso'+ '_file'+str(ifile+5)+'_seuil'+str(iseuil)+'.png')
        plt.close()
            
            
            
        
        
        
    ifile = ifile+10
#    ifile+=10
    
    
np.save(outpathNPY+myfile[-46:-4]+'_iso'+'_seuils'+'.npy', matrix)


#grid = griddata((zrON[:,0], zrON[:,1]), zrON[:,2], zrNANxy, method='linear')   
#   
#matrix = np.zeros(zr.shape)
#matrix[zrNANxy[:,0],zrNANxy[:,1]]= grid
#matrix[zrON[:,0].astype(int),zrON[:,1].astype(int)]=zrON[:,2]
   
   # Ecrase les valeurs interpolées de la terre par le masque terre (en np.nan).
#matrix = matrix+landmask



print 'end'


#============================
# # *** Dictionnaire ***
#============================

#plt.plot(zrcontour.allsegs)
#zrcontour.__dict__
#zrcontour.collections[0].get_paths()

#p = zrcontourf.collections[0].get_paths()[0]
#v = p.vertices
#x = v[:,0]
#y = v[:,1]

#**
#r.round(1)

#    A = ndimage.binary_dilation(r).astype(r.dtype)
#    contours de zr avec remplissage
#    1
#    contours de zr sans remplissage
#    zrcontour = plt.contour(r,seuils, origin='upper')
#    ouverture du contourf
#    A = ndimage.binary_opening(zrcontour)
#    plt.imshow(zrcontour, norm=norm_chl, origin='upper', cmap=new_map_chl,) 

#fig2 = plt.gcf()
##        plt.imshow(matrix[ifile,iseuil])
#rcontour = plt.contour(result2, 1, origin='upper')
#plt.imshow(rcontour)
#fig2.savefig(outpathPNG+myfile[-46:-4]+'_iso'+'_SeuilContour'+str(iseuil)+'.png')

# Figures sur plusieurs axes.
#        fig, (ax1, ax2) = plt.subplots(1,2)
##    plt.imshow(zr_conv, norm=norm_chl, origin='upper', cmap=new_map_chl,)
#        ax1.imshow(zrs, norm=norm_chl, origin='upper', cmap=new_map_chl,)
#        ax2.imshow(zre1, norm=norm_chl, origin='upper', cmap=new_map_chl,) 
##    ax3.imshow(zr_convfft, norm=norm_chl, origin='upper', cmap=new_map_chl,) 
#        plt.show()
#        fig.savefig(outpathPNG+myfile[-46:-4]+'_iso'+'seuil'+'.png', dpi=200, bbox_inches='tight')

#a = np.array([[0, 0, 0, 0, 0],
#              [0, 0, 1, 1, 0],
#              [0, 0, 1, 1, 0],
#              [0, 0, 0, 0, 0],
#              [0, 0, 0, 0, 0]])  
#              
#b = np.array([[0, 0, 0, 0, 0],
#              [0, 0, 0, 0, 0],
#              [0, 0, 1, 1, 0],
#              [0, 0, 1, 1, 0],
#              [0, 0, 1, 1, 0]])  
#
#i = np.array([[0, 0, 0, 0, 0],
#              [0, 0, 0, 0, 0],
#              [0, 0, 0, 0, 0],
#              [0, 0, 0, 0, 0],
#              [0, 0, 0, 0, 0]])  

