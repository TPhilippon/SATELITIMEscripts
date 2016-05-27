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
    outpathPNG2 = homepath+'\\SATELITIME\\data\\contours\\iso_png2\\'
else : 
    path = homepath+'/SATELITIME/data/contours/interp_npy/'
    outpathNPY = homepath+'/SATELITIME/data/contours/iso_npy/'
    outpathPNG = homepath+'/SATELITIME/data/contours/iso_png/'
    outpathPNG2 = homepath+'/SATELITIME/data/contours/iso_png2/'
    
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
# #                         Morpho & interp 2D
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
        SEUIL = int(seuils[iseuil]*100)
        fig1.savefig(outpathPNG2+myfile[-52:-4]+'_iso'+'_seuil'+str(format(SEUIL,'03'))+'_file'+str(format(ifile,'02'))+'.png')

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
i = 80

#while ifile <= i:  #Files loop
for myfile in data:
    
    for iseuil in range(7):   #Concentration seuils loop
        
        print 'ifile =', ifile+5, '   iseuil =', iseuil
        
        #--------- Alternate --------
       
        mat = np.copy(matrix)
        
        d1 = mat[ifile,iseuil,:,:]
        d2 = mat[ifile+10,iseuil,:,:]
        
        d1[d1==1]=99
        d1[d1==0]=2
        d1[d1==99]=0
        
        d2[d2==1]=99
        d2[d2==0]=3
        d2[d2==99]=0
        
        # (-3)--> d1 change. (2) --> d2 change. (0) --> no change (plume). (-1) --> no change (else)
        d12 = d1 - d2
        
        
        # --- d12 construction du format x, y ,z ---
        # valeurs d'interpolation pour d1
        d12[d12==0] = 1
        d12[d12==-1] = 0
    
#        xyd1 = np.asarray(np.where(d12==-3)).T
#        xyd2 = np.asarray(np.where(d12==2)).T
#        xyd12 = np.asarray(np.where(~d12!=-3)).T
        
        x12=np.indices(d12.shape)[0]
        y12=np.indices(d12.shape)[1]
        x12=x12.flatten()
        y12=y12.flatten()
        m12=d12.flatten()        
        
        mat312 = np.vstack((x12,y12))
        mat312 = np.vstack((mat312,m12))
        mat312 = mat312.T
        
        mat312INT = np.copy(mat312)
        
        coordsd1INT = mat312INT[mat312INT[:,2]==-3]
        coordsd2INT = mat312INT[mat312INT[:,2]==2]
#        coordsINT = mat3[np.isnan(mat3[:,2])]
        
        mat312s1 = np.copy(mat312)
        mat312s2 = np.copy(mat312)        
        
        #Valeurs d'interpolation pour d1
        coordsd1 = mat312s1[mat312s1[:,2]!=-3]
        coordsd1[:,2][coordsd1[:,2]==2] = 0
        #Valeurs d'interpolation pour d2. On veut le gradient inverse à d1.
        coordsd2 = mat312s2[mat312s2[:,2]!=2]
        coordsd2[:,2][coordsd2[:,2]==-3] = 0
        coordsd2[:,2][coordsd2[:,2]==1] = 99
        coordsd2[:,2][coordsd2[:,2]==0] = 1
        coordsd2[:,2][coordsd2[:,2]==99] = 0

        
        print 'interpolation d1 --->' 
#        interpd1 = griddata((coordsd1[:,0], coordsd1[:,1]), coordsd1[:,2], coordsd1INT[:,0:2], method='linear')
        interpd1 = griddata((coordsd1[:,0], coordsd1[:,1]), coordsd1[:,2], coordsd1INT[:,0:2], method='linear')

        
        print 'interpolation d2 --->'
        interpd2 = griddata((coordsd2[:,0], coordsd2[:,1]), coordsd2[:,2], coordsd2INT[:,0:2], method='linear')        
        
#        matd1 = np.zeros(d1.shape)
#        matd1[coordsd1INT[:,0].astype(int),coordsd1INT[:,1].astype(int)]= interpd1
#        matd1[coordsd1[:,0].astype(int),coordsd1[:,1].astype(int)]=coordsd1[:,2]
        
        # ***
#        matd12 = np.zeros(d1.shape)
#        matd12[coordsd1[:,0].astype(int),coordsd1[:,1].astype(int)]=coordsd1[:,2]
#        matd12[coordsd2[:,0].astype(int),coordsd2[:,1].astype(int)]=coordsd2[:,2]
#        matd12[coordsd1INT[:,0].astype(int),coordsd1INT[:,1].astype(int)]= interpd1
#        matd12[coordsd2INT[:,0].astype(int),coordsd2INT[:,1].astype(int)]= interpd2
        # ***
        
        time = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        matALL = np.zeros(d1.shape)
        
        for t in range (0,9):   #4D seuils loop
            print '4d stacking ....  '+str(time[t])            
            
            matdXa = 'matd12a'+str(i)
            matdXb = 'matd12b'+str(i)
            matdXa, matdXb  = np.zeros(d1.shape), np.zeros(d1.shape)
            
            #Set commun values in the arrays.
            matdXa[coordsd1[:,0].astype(int),coordsd1[:,1].astype(int)]=coordsd1[:,2]
            matdXb[coordsd2[:,0].astype(int),coordsd2[:,1].astype(int)]=coordsd2[:,2]
            
            #Set new interpolated values in the arrays.
            matdXa[coordsd1INT[:,0].astype(int),coordsd1INT[:,1].astype(int)]= interpd1
            matdXb[coordsd2INT[:,0].astype(int),coordsd2INT[:,1].astype(int)]= interpd2
            
            #Select d1 and d2 values for time[t].
            a = np.zeros(d1.shape)
            b = np.zeros(d1.shape)
            
            a = a+(matdXa>time[t])
            b = b+(matdXb<time[t])
#            matdXb[matdXb<time[t]]=-1
#            matdXb[matdXb>=time[t]]=0
#            matdXb[matdXb==-1]=1
#            matdXa[matdXa>time[t]]=10-time[t]
#            matdXa[matdXa<=time[t]]=0
            #Complie d1 and d2 values in binary (0,1) matrix.
            matd12 = a + b
            matd12[matd12==2]=1
            #Conversion of values=2 to 1, to get only 1 or 0 values.
#            matd12[matd12==2]=1
            
            matrix[ifile+t+1, iseuil] = matd12
            matALL = matALL+matd12
            
            fig3 = plt.gcf()

            plt.imshow(matd12)
            SEUIL = int(seuils[iseuil]*100)
            fig3.savefig(outpathPNG2+myfile[-52:-4]+'_iso'+'_seuil'+str(format(SEUIL,'03'))+'_file'+str(format(ifile+t+1, '02'))+'.png')
            plt.close()
            

        
#        matd122 = np.zeros(d1.shape)
#        matd122[coordsd1INT[:,0].astype(int),coordsd1INT[:,1].astype(int)]= interpd1
#        matd122[coordsd2INT[:,0].astype(int),coordsd2INT[:,1].astype(int)]= interpd2
#        
#        matd123 = np.zeros(d1.shape)
#        matd123[coordsd1INT[:,0].astype(int),coordsd1INT[:,1].astype(int)]= interpd1
#        matd123[coordsd2INT[:,0].astype(int),coordsd2INT[:,1].astype(int)]= interpd2
     
        
        
#        matrix[ifile+5, iseuil] = matd12
        
        #-----------------------------
#        Saving numpy matrix.
    fig2 = plt.gcf()

    plt.imshow(matALL)
    SEUIL = int(seuils[iseuil]*100)
    fig2.savefig(outpathPNG2+myfile[-52:-4]+'_iso'+'_seuil'+str(format(SEUIL,'03'))+'_file'+str(format(ifile,'02'))+'.png')
    plt.close()
            
            
    ifile = ifile+10
#    ifile+=10
    
    
np.save(outpathNPY+myfile[-52:-4]+'_morpho'+'_seuils'+'_interp4d'+'.npy', matrix)





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

