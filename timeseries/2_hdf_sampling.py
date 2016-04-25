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
from pyhdf.SD import SD, SDC   
from pylab import mpl as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import time

# ============================================================================
# =========================== Variables parametrables ========================
# ============================================================================
if os.path.exists('/home/pressions'):
    prefix='/home/'
else:
    prefix='/media/'            
prefix=prefix+'pressions/'
interpolation='yes' # 'no' 'yes"
bzi=40    # Buffer autour de la zi pour l'affichage                                            # marge autour des ZI pour le zoom
varnum = 3                                                              # varnum pour 1 seule variable
sensor = 'aqua' #aqua swf'  # 'swf'                                                        # swf ou aqua
reso = '9km'  # '4km' ou '9km'  
imformat='png' # or tif'                                                        # résolution '4km', '4', '9km' ou '9'.
#listvariable = ['nsst_8d', 'poc_8d', 'sst11mic_8d', 'chl_8d','pic_8d']      # variables 8D
listvariable = ['nsst_32d', 'poc_32d', 'sst11mic_32d', 'chl_32d','pic_32d','sst11mic_32d','GIOP_adg_32d']   #[varnum variables 32D

nzi=4                   # nombre de zi
landmask=prefix+'SATELITIME/data/landmaskZRcircle.png'
imlandmask=plt.imread(landmask)

# =========================================================================
# =========================== Definition Varaibles ========================
# =========================================================================



if interpolation == 'yes':
    interp='interpn'
else:
    interp='no_interp'
    
scaling=[[21,35,'linear'],[10,1000,'log'],[21,35,'linear'],[0.005,20,'log'],[0.00001,0.05,'log'],[21,35,'linear'],[0.001,1,'log']]
fillvalue=[65535.0,-32767.0,65535.0,-32767.0,-32767.0,65535.0,-32767.0][varnum]                          # Fill values pour chaque variable
slI=[(0.00071718,-2),(1.0,0),(0.00071718,-2),(1.0,0),(1.0,0),(0.00071718,-2),(1.0,0)]                       # Pentes et intercept pour 'nsst_8d', 'poc_8d', 'sst11mic_8d', 'chl_8d'
variable = listvariable[varnum]
slope=slI[varnum][0]                                                    # égal au premier de la paire
intercept=slI[varnum][1] 
vmin=scaling[varnum][0]
vmax=scaling[varnum][1]
scalingtype=scaling[varnum][2]



ezi=np.zeros((nzi,),dtype=('i4,i4,i4,i4'))  

                            # Stockage des coordonnées des ZI
imfile = prefix+'SATELITIME/data/Chl2009_'+reso+'.hdf'          # Image pour voir les pixels (choisir la bonne résolution)
mois=['janv','fev','mars','avr','mai','juin','juill','aout','sept','oct','nov','dec']
mois=mois*20                # Pour couvrir 13 années
unit=[u'Temperature de surface en °C',u'Carbone organique particlaire (POC) en mg.m-3',u'Temperature de surfarce nocturne en °C',u'Chlorophylle en mg.m-3',u'Particulate Inorganic Carbon en mol.m-3'u'Chlorophylle en mg.m-3',u'Sea surface temperature in °C',u'Absorption due gelbstoff and detrital material at 443 nm, GIOP model'][varnum]




# =========================================================================
# =========================== Definition PATHS ============================
# =========================================================================

pathZR =prefix+'SATELITIME/data/ZR/'+sensor+'/'+str(variable)+'/'+reso+'/'
pathZI =prefix+'SATELITIME/data/ZI/'       # "+sensor+'/'+str(variable)+'/'+reso+'/'                       
data_in =prefix+'SATELITIME/data/FULL/'+str(variable)+'/'+sensor+'/'+reso+'/'
pathPNG = prefix+'SATELITIME/data/PNG/'+sensor+'/'+str(variable)+'/'+reso+'/png_caraibe/'+interp+'/'
if not os.path.exists(pathPNG):
    os.makedirs(pathPNG)


filezi = pathZI+sensor+'_'+str(variable)+reso+'_ZIs.npy'

# =========================================================================
# =========================== Creation Matrice Vide des ZIs ===============
# =========================================================================
files = os.listdir(data_in)                     #  Liste les fichiers.
files.sort()    
nfiles=len(files)    

filename, file_extension = os.path.splitext(files[0])

print '=================================='
print variable
print 'sensor =       '+reso
print 'fillvalue=     '+str(fillvalue)
print 'slope =        '+str(slope)
print 'vmin =         ' +str(vmin)
print 'vmax =         ' +str(vmax)
print 'scaling type = '+str(scalingtype)
print 'intercept =    '+str(intercept)
print 'interpolation ='+interpolation
print 'unit =          '+unit
print 'nombre de fichiers= ', nfiles
print 'fichier 0 = '+files[0]
print 'fichier 1 = '+files[1]  
print 'file extension  = '+ file_extension

print "===================================="
raw_input('Press any key to proceed')

#j=len(ZIs)/2                        # Nombre de ZI (1er element de [ZIs] = coord, 2e element = data)
expr=[]                             # Liste pour former la matrice
for numzi in range(0,nzi):
    expr=expr+[('zi'+str(numzi)+'n','f8'),('zi'+str(numzi)+'moy','f8'),('zi'+str(numzi)+'et','f8'),('zi'+str(numzi)+'min','f8'),('zi'+str(numzi)+'max','f8'),('zi'+str(numzi)+'nan','f8')]
expr=[('date','i8')]+expr    
print expr
ZIs = np.zeros((nfiles,),dtype=expr)     # matrice vide à remplir de taille 'expr' fois nfiles
        
# =========================================================================
# Definition de la Zone Regionale : ZR et des zones d'interets -----------------------------
# ---- Sampling Guadeloupe, Martinique, Iles du Nord, Large : xmin, xmax, ymin, ymax -----
# =========================================================================

if reso=='9km':
    xzrmin,xzrmax,yzrmin,yzrmax=1250,1750,750,1100   # Caraibes 
    ezi[0:nzi]=([(1430,1433,882,888),(1433,1437,901,907),(1410,1412,861,864),(1710,1717,897,902)])          # Coordonnées des ZI 9km (ajustées)
if reso=='4km':
    xzrmin,xzrmax,yzrmin,yzrmax=2500,3500,1500,2200
    ezi[0:nzi]=([(2860,2866,1764,1777),(2867,2874,1802,1814),(2820,2825,1722,1728),(3420,3434,1795,1805)])  # Coordonnées des ZI 4km (ajustées)

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


# ===================================================================================
# ======================== zone régionale de reference ==============================
# ===================================================================================

# LA ZR est affiché en utilisant un image annuelle de la Chl
                          
fzr = SD(imfile, SDC.READ)      #  Lire depuis le hdf.
l3m = fzr.select('l3m_data')   #  choisit le hdf datasets
l3m = l3m.get()               #  Fonction get() pour avoir vraiment le tableau pour lire le hdf.
#l3m = np.dot(l3m,1.0)
zrref=l3m[yzrmin:yzrmax,xzrmin:xzrmax]
plt.imshow(zrref,norm=norm_chl,extent=[xzrmin,xzrmax,yzrmin,yzrmax],origin='lower', cmap=new_map_chl,interpolation='none',aspect='equal') #
plt.gca().invert_yaxis()

plt.show()

# =======================================================================
# =========================== zone interet ==============================
# =======================================================================

for i in range(0,ezi.size):
    print "ZI numero :",i
    xzimin,xzimax,yzimin,yzimax=ezi[i][0],ezi[i][1],ezi[i][2],ezi[i][3]
    zib=l3m[yzimin-bzi:yzimax+bzi,xzimin-bzi:xzimax+bzi]
    zi=l3m[yzimin:yzimax,xzimin:xzimax]
    plt.imshow(zib,norm=norm_chl,extent=[xzimin-bzi,xzimax+bzi,yzimin-bzi,yzimax+bzi],origin='lower', cmap=new_map_gray_chl,interpolation='none',aspect='equal') #
    plt.imshow(zi,norm=norm_chl,extent=[xzimin,xzimax,yzimin,yzimax],origin='lower', cmap=new_map_chl,interpolation='none',aspect='equal') 
    plt.gca().invert_yaxis()
    plt.autoscale()

    plt.show()
    plt.close()
    
    


bakmap=zrref
bakmap[ bakmap != -32767.] = 0
bakmap[ bakmap == -32767.] = np.nan

# =======================================================================
# ====================EXTRACTION DES DONNEES==============================
# =======================================================================

fig=plt.figure()
ax=fig.add_subplot(111)
#plt.ion()
#plt.show()
  


print data_in

counter=0    
for myfile in files:

    pngfig=pathPNG+myfile[0:41]+'_'+interp+'.'+imformat 
    print pngfig
    if  not os.path.exists(pngfig):    #not 
        
        date=myfile[1:15]
        annee = myfile[1:5]  
        j = int(myfile[5:8])
        j=int(j)
        if j >= 355 or j < 83:
            saison = u'Hiver'
        if j >= 83 and j < 176:
            saison = u'Printemps'
        if j >= 176 and j < 261:
            saison = u'Eté'
        if j >= 261 and j < 355:
            saison = u'Automne'    
        i = int(j / 30.5)                # Arrondi (interger)
        if i >= 12:                         # 0 = janvier et 11 = décembre
            i = 11
        title=saison+'-'+str(mois[int(i)])+'-'+str(annee)
    
    
        
        print 'reading', myfile,date
        start_time=time.time()
        if file_extension== '.hdf' :
            
            # if HDF data =================================        
    
            File = SD(data_in+myfile, SDC.READ)         #  Lire depuis le hdf.
            data= File.select('l3m_data')               #  Et met le contenu dans File.
            data = data.get()
            data = data.astype('float') 
            data[ data == fillvalue ] = np.nan
            
            # if NetCFD4 data =============================        
            
        if file_extension == '.nc':
            File = Dataset(data_in+myfile, 'r', format='NETCDF4')
            data=File['adg_443_giop']
            data=data[:]
            data=data.data
            #data=data[1]
            #sys.exit() 

            #data = data.astype('float') 
            data[ data == fillvalue ] = np.nan





        data=(data*slope)+intercept           
        zr=data[yzrmin:yzrmax,xzrmin:xzrmax]
        print("--- %s seconds ---" % (time.time()-start_time))
    
        # =======================================================================
        # ====================INTERPOLATOR==============================
        # =======================================================================
    
        #zr4i = zr
        if interpolation == 'yes':
            # a boolean array of (width, height) which False where there are missing values and True where there are valid (non-missing) values
            mask = np.isnan(zr)
            mymask=mask    
            mask=~mask
            # array of (number of points, 2) containing the x,y coordinates of the valid values only
            xx, yy = np.meshgrid(np.arange(zr.shape[1]), np.arange(zr.shape[0]))
            xym = np.vstack( (np.ravel(xx[mask]), np.ravel(yy[mask])) ).T
            # the valid values in the channel,  as 1D arrays (in the same order as their coordinates in xym)
            zr = np.ravel( zr[mask] )
            
            #interp0 = scipy.interpolate.interpn( xym, zr )        
            #interp0 = scipy.interpolate.CloughTocher2DInterpolator( xym, zr )
            print 'interpolating'            
            interp0 = scipy.interpolate.NearestNDInterpolator( xym, zr )
            
            # interpolate the whole image, 
            result0 = interp0(np.ravel(xx), np.ravel(yy)).reshape( xx.shape )
            zr=result0
            #zr = zr+bakmap # Masque continent   
        
        # =======================================================================
        # ====================PREPARE FIGURE PNG==============================
        # =======================================================================


        plt.title(title+' \n '+myfile)    
        im=plt.imshow(zr,norm=norm, origin='upper', cmap=new_map,)
        plt.imshow(imlandmask)
        plt.axis('off')
        
        start_time=time.time()
        if counter >-10 : 
            divider=make_axes_locatable(plt.gca())
            cax=divider.append_axes("right","5%",pad="1%")
            cbar=plt.colorbar(im,cax)
            cbar.ax.set_ylabel(unit)
        
        pngfig=pathPNG+myfile[0:41]+'_'+interp+'.'+imformat   
        plt.tight_layout()
        if not os.path.exists(pngfig):
            plt.savefig(pngfig,format=imformat,dpi=200,bbox_inches='tight')
        plt.pause(0.0001)        
        print("--- %s seconds ---" % (time.time()-start_time))
#    
        filezr = pathZR+interp+'/'+myfile[0:39]+'_ZR.npy'
        print 'saving ',filezr
                
        np.save(filezr,zr)
        #plt.clf()
        #sys.exit()
        
        # =======================================================================
        # ====================SAVE ZIS==============================
        # =======================================================================    
        
        #ZIs=[]
        dline=[] 
        for i in range(0,nzi):
            xzimin,xzimax,yzimin,yzimax=ezi[i][0],ezi[i][1],ezi[i][2],ezi[i][3]
            zi=data[yzimin:yzimax,xzimin:xzimax]
            #ZIsZIs=ZIs+[(xzimin,xzimax,yzimin,yzimax),zi]      
            zi_num=zi.size # nombre de pixels
            zi_mean = np.nanmean(zi)
            zi_nanmin=np.nanmin(zi)
            zi_nanmax=np.nanmax(zi)
            zi_nanstd=np.nanstd(zi) 
            zi_nan=np.isnan(zi).sum()
            dline=dline+[zi_num,zi_mean,zi_nanstd,zi_nanmin,zi_nanmax,zi_nan]
        dline=[date]+dline
        ZIs[counter]=tuple(dline)  
        #sys.exit()                                                     ### <-- sys.exit pour tester
                                               ### SAVE
        plt.imshow(np.zeros(np.shape(zr)),cmap=plt.get_cmap('Greys'))
        plt.close()
        counter=counter+1

np.save(filezi,ZIs) # 
np.savetxt(filezi+'.csv',ZIs,delimiter=',',comments='',header=','.join(ZIs.dtype.names),newline='\n')

print 'Fin'


