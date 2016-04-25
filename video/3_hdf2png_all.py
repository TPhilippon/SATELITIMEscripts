#!/usr/bin/env python
# -*- coding: utf8 -*-



import os
import sys
import numpy
from pyhdf.SD import SD, SDC
from mpl_toolkits.basemap import Basemap, cm
from matplotlib.colors import LogNorm

from pylab import *
import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.signal


#---------- read data ----------#

# Suivant : pour pouvoir convertir.
varnum=1  # on a dl plusieurs variables (ligne 23)
varg=['sst11mic_8d','poc_8d', 'nsst_8d','chl_8d'][varnum]
title=[u'Temperature de surface en °C',u'Carbone organique particlaire (POC) en mg.m-3',u'Temperature de surfarce nocturne en °C',u'Chlorophylle en mg.m-3'][varnum]
FillValue=[65535.0,-32737.0,-32737.0,-32737.0]
slI=[(0.00071718,-2),(1,0),(0.00071718,-2),(1,0)][varnum] # cette fois varnum récupère pente et intercept. 
slope=slI[0] # égal au premier de la paire
intercept=slI[1] 

#:Scaling = "linear" ;
#:Scaling_Equation = "(Slope*l3m_data) + Intercept = Parameter value" ;
#:Slope = 0.15000001f ;
#:Intercept = -2.f ;
#:Data_Minimum = -1.799999f ;
#:Data_Maximum = 35.25f;

#ScaledDataMinimum= [-2,10,-2,0.01][varnum] # Nous sert dans la formule de convertion. Scaled = données mises à l'échelle.
#ScaledDataMaximum= [45,1000,45,20][varnum] # Donne les futurs min / max des unités de valeurs.
#norm=[mpl.colors.Normalize(vmin=ScaledDataMinimum, vmax=ScaledDataMaximum),mpl.colors.LogNorm(vmin=ScaledDataMinimum,vmax=ScaledDataMaximum), \
#      mpl.colors.Normalize(vmin=ScaledDataMinimum, vmax=ScaledDataMaximum),mpl.colors.LogNorm(vmin=ScaledDataMinimum, vmax=ScaledDataMaximum)][varnum]
path = varg+'/hdf/'
files = os.listdir(path) #Liste ldes fichiers.
files.sort() #Trie les fichiers.
print len(files) #len = longueur de la liste de fichiers.

ymin=1002
ymax=1449
xmin=4086
xmax=5212

latmax=90-ymin*0.04166666
latmin=90-ymax*0.04166666
longmin=xmin*0.04166667-179.9792
longmax=xmax*0.04166667-179.9792

#-------------------------------------------------------

for myfile in files:

    annee = myfile[1:5]
    j = int(myfile[12:15])
    j=int(j)
 
    if j >= 355 or j < 83:
        saison = u'Hiver'
    if j >= 83 and j < 176:
        saison = u'Printemps'
    if j >= 176 and j < 261:
        saison = u'Eté'
    if j >= 261 and j < 355:
        saison = u'Automne'

    File_Name = myfile
    File = SD(varg+'/hdf/'+File_Name, SDC.READ) #   Lire depuis le hdf.
    l3 = File.select('l3m_data') #  Et met le contenu dans File.
    l3d = l3.get() #    Fonction get() pour avoir vraiment le tableau pour lire le hdf.
    

##    print l3d.shape
##    print ymin,ymax
##    print latmin,latmax
##    print
##    print xmin,xmax
##    print longmin,longmax
##    print
    
    print annee,j,l3d.min(),l3d.max() # On peut demander les valeurs min / max.

    l3d=l3d[1002:1449,4086:5212] # Echantilloner la mediterranee. Distance lignes puis colonnes.
    l3d=np.dot(l3d,1.0) #   CRUCIAL : transformation en objet numpy pour manipuler plus facilement.
    

    vmin=ScaledDataMinimum+2
    vmax=ScaledDataMaximum

    l3d[ (l3d < vmin) & (l3d != FillValue) ] = vmin #
    #l3d[ l3d > 10 ] = 10.0 #
    
    l3d[ l3d == FillValue ] = np.nan # 0.011 #0.00001 #
    print annee,j,l3d.min(),l3d.max()



    #------------ Interpolation


    print 'interpolating....'
    data = l3d

    # a boolean array of (width, height) which False where there are missing values and True where there are valid (non-missing) values
    mask = np.isnan(l3d)
    mask=~mask
    # array of (number of points, 2) containing the x,y coordinates of the valid values only
    xx, yy = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
    xym = np.vstack( (np.ravel(xx[mask]), np.ravel(yy[mask])) ).T
    # the valid values in the first, second, third color channel,  as 1D arrays (in the same order as their coordinates in xym)
    data = np.ravel( data[mask] )
    # three separate interpolators for the separate color channels
    interp0 = scipy.interpolate.NearestNDInterpolator( xym, data )
    # interpolate the whole image, one color channel at a time    
    result0 = interp0(np.ravel(xx), np.ravel(yy)).reshape( xx.shape )
    l3d=result0
    l3d=l3d*slope+intercept

    print 'end interp'

    
    #---------- Plot Data Global Map ----------#

    m = Basemap(projection='cyl',llcrnrlat=latmin,urcrnrlat=latmax, llcrnrlon=longmin, \
               urcrnrlon=longmax,resolution='c')
    colors = [(0.33,0.33,0.33)] + [(plt.cm.jet(i)) for i in xrange(1,256)]
    new_map = matplotlib.colors.LinearSegmentedColormap.from_list('new_map', colors, N=256) # Colormap



    img= m.imshow(np.flipud(l3d), norm=norm, cmap=new_map , interpolation='bilinear') # plt.get_cmap('gray') #vmin=ScaledDataMinimum, vmax=ScaledDataMaximum,
    m.drawlsmask(land_color='beige',ocean_color='None',lakes=True)
    cb = m.colorbar(img,"bottom", size="5%", pad='4%')
    cb.set_label(title)

    plt.title(saison+'   '+annee+'   jour : '+str(j))
    plt.savefig(varg+'/png/'+File_Name+'.png',dpi=200,bbox_inches='tight')
    
     #plt.show()
    pause('Pausing for click')
    plt.close() 
