## -*- coding: utf-8 -*-
"""
Created on Thu May 28 14:32:07 2015
@author: Pierre-Alain Mannoni, Térence Philippon
"""
# Email : Pierre-Alain.MANNONI@unice.fr & terence.philippon@sfr.fr

#####          ** Affichage ZR et ZIs avant extraction **
#       ** Changer les variables directement et/ou avec varnum **
#################################################################################
# Extraction par zones (ZI / ZR) des données du site oceandata.sci.gsfc.nasa.gov
# ZI : zones d'intérêts, en général de petite taille pour traitement graphique. 
# ZR : zone régionale, de taille plus grande pour générer des .png.
# Remarques :
#Les coordonnées des ZR et ZI peuvent être modifiées dans le script.  
# ===============================================================================
# ============= Imports =========================================================

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


# ============================================================================
# =========================== Variables parametrables ========================
# ============================================================================
varnum = 3
reso='9km'
fname= 'SATELITIME/data/ZR/aqua/chl_32d/9km/interpn/A20103292010360.L3m_R32_CHL_chlor_a_9km_ZR.npy'
fname= 'SATELITIME/data/ZR/aqua/chl_32d/9km/interpn/A20023052002336.L3m_R32_CHL_chlor_a_9km_ZR.npy'


if os.path.exists('/home/pressions'):
    prefix='/home/'
else:
    prefix='/media/'            
prefix=prefix+'pressions/'


listvariable = ['nsst_32d', 'poc_32d', 'sst11mic_32d', 'chl_32d','pic_32d','sst11mic_32d','GIOP_adg_32d']   #[varnum variables 32D
landmask=prefix+'SATELITIME/data/landmaskZRcircle.png'
imlandmask=plt.imread(landmask)

# =========================================================================
# =========================== Definition Varaibles ========================
# =========================================================================

imformat='png' # or tif' 
    
scaling=[[21,35,'linear'],[10,1000,'log'],[21,35,'linear'],[0.005,20,'log'],[0.00001,0.05,'log'],[21,35,'linear'],[0.001,1,'log']]
fillvalue=[65535.0,-32767.0,65535.0,-32767.0,-32767.0,65535.0,-32767.0][varnum]                          # Fill values pour chaque variable
slI=[(0.00071718,-2),(1.0,0),(0.00071718,-2),(1.0,0),(1.0,0),(0.00071718,-2),(1.0,0)]                       # Pentes et intercept pour 'nsst_8d', 'poc_8d', 'sst11mic_8d', 'chl_8d'
variable = listvariable[varnum]
slope=slI[varnum][0]                                                    # égal au premier de la paire
intercept=slI[varnum][1] 
vmin=scaling[varnum][0]
vmax=scaling[varnum][1]
scalingtype=scaling[varnum][2]


mois=['jan','feb','mar','apr','may','jun','jul','aug','sept','oct','nov','dec']
imfile = prefix+'SATELITIME/data/Chl2009_'+reso+'.hdf'          # Image pour voir les pixels (choisir la bonne résolution)
mois=mois*20                # Pour couvrir 13 années
unit=[u'Temperature de surface en °C',u'Carbone organique particlaire (POC) en mg.m-3',u'Temperature de surfarce nocturne en °C',u'Chlorophylle en mg.m-3',u'Particulate Inorganic Carbon en mol.m-3'u'Chlorophylle en mg.m-3',u'Sea surface temperature in °C',u'Absorption due gelbstoff and detrital material at 443 nm, GIOP model'][varnum]


# Colormap Chl de ref
norm_chl=mpl.colors.LogNorm(vmin=0.01, vmax=20)
colors = [(0.33,0.33,0.33)] + [(plt.cm.jet(i)) for i in xrange(1,256)]
new_map_chl = mpl.colors.LinearSegmentedColormap.from_list('new_map_chl', colors, N=256) 
colors = [(0.33,0.33,0.33)] + [(plt.cm.gray(i)) for i in xrange(1,256)]
new_map_gray_chl = mpl.colors.LinearSegmentedColormap.from_list('new_map_gray_chl', colors, N=256) 

# Colormap Couleur et grise
if scalingtype=='log':norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
if scalingtype=='linear':norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax)
colors = [(0.33,0.33,0.33)] + [(plt.cm.jet(i)) for i in xrange(1,256)]
new_map = mpl.colors.LinearSegmentedColormap.from_list('new_map', colors, N=256) 
colors = [(0.33,0.33,0.33)] + [(plt.cm.gray(i)) for i in xrange(1,256)]
new_map_gray = mpl.colors.LinearSegmentedColormap.from_list('new_map_gray', colors, N=256) 


mois=['jan','feb','mar','apr','may','jun','jul','aug','sept','oct','nov','dec']
# =========================================================================
# =========================== Definition PATHS ============================
# =========================================================================
path=prefix+'SATELITIME/data/ZR/aqua/chl_32d/9km/interpn/'
savepath=prefix+'SATELITIME/data/PNG/aqua/chl_32d/9km/png_caraibe/interp_iso/'
listing = glob.glob(path+'*.npy')
listing=sort(listing)

# =======================================================================
# ====================PREPARE FIGURE PNG==============================
# =======================================================================

print 'Starting...'
print path

for myfile in listing:

    
    
    date=myfile[-45:-31]
    annee = date[0:4]  
    j = int(date[4:7])
    j=int(j)
    if j >= 355 or j < 83:
        saison = u'Winter'
    if j >= 83 and j < 176:
        saison = u'Spring'
    if j >= 176 and j < 261:
        saison = u'Summer'
    if j >= 261 and j < 355:
        saison = u'Autumn'    
    i = int(j / 30.5)                # Arrondi (interger)
    if i >= 12:                         # 0 = janvier et 11 = décembre
        i = 11
    title=saison+'-'+str(mois[int(i)])+'-'+str(annee)
    
    
        
    print 'reading', title
    
    
    
    print myfile 
    zr=np.load(myfile)
    fname=os.path.basename(myfile)    
    #plt.title(title+' \n '+myfile)    
    plt.imshow(imlandmask)    
    im=plt.imshow(zr,norm=norm, origin='upper', cmap=new_map,)
    plt.imshow(imlandmask)

    # ======== Contour Line ===============
    plt.contour(zr,[0.2])   
    plt.title(title)
    
    plt.axis('off')


    
    pngfig=savepath+myfile[-46:-4]+'_iso.'+imformat   
    if not os.path.exists(pngfig):
        plt.savefig(pngfig,format='png',dpi=72,bbox_inches='tight')
    #plt.show() 
    #raw_input(date)
    plt.clf()    
    #plt.close('all')
