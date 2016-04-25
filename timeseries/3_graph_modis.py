# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 09:49:45 2015

@author: Pierre-Alain Mannoni, Térence Philippon
"""
# Email : Pierre-Alain.MANNONI@unice.fr & terence.philippon@sfr.fr

# ** Pour MODISA et SWF : Traitement mathématique et save en .npy **
# ** NbTot de data, Moyenne, Ecart-type, mini/max, nb de NAN **
# ** utilisation des fichiers ZI créés préalablement **
#################################################################################
# Plot graphique après le stockage des valeurs recherchées dans une matrice.
# Valable en 8 et 32D (vérifier la concordance des lignes commentées/activées)
# 
# Remarques :
# La mise à l'échelle de l'axe des x en fin de script est modifiable suivant les
# données utilisées (nb de fichiers pour longueur axe) et l'affichage souhaité.
# ===============================================================================
# =========================== Imports ===========================================

import os
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import glob

#--------------------- Définition variables et directory ------------------------

if os.path.exists('/home/pressions'):
    prefix='/home/'
else:
    prefix='/media/'            
    
path=prefix+'pressions/SATELITIME/data/ZI/npy/'
savepath=prefix+'pressions/SATELITIME/data/PNG/timeplot/'

listing = glob.glob(path+'*.npy')
listing=sort(listing)
print listing
print 'Starting...'
counter=0
for file in listing:
    
    #print file 
    ZIs=np.load(file)
    fname=os.path.basename(file)
    
    
    # -------------------- Bornes mois ou saisons  ----------------------------------
    
    i =0
    date=[]
    mois=['janv','fev','mars','avr','mai','juin','juill','aout','sept','oct','nov','dec']
    mois=mois*13                # Pour couvrir 13 années
    #for myfile in files :       # Boucle : Arrondi de la date au mois. 
    #    print '..........',myfile
    #    annee = myfile[1:5]     
    #    j = myfile[5:8]
    #    jour=int(j)
    #    i = int(jour / 30.5)                # Arrondi (interger)
    #    if i >= 12:                         # 0 = janvier et 11 = décembre
    #        i = 11
    #
    #    date=date+[str(mois[int(i)]+'-'+str(annee))]
    date=ZIs['date']
    # ------------------------- Plot graphique --------------------------------------
    
    
    
    
    
    plt.plot(ZIs['zi0moy'],color='r', linestyle='-', label='Guadeloupe')        
    plt.plot(ZIs['zi1moy'],color='b', linestyle='-', label='Martinique')        
    plt.plot(ZIs['zi2moy'],color='g', linestyle='-', label='Iles_du_Nord')       
    plt.plot(ZIs['zi3moy'],color='y', linestyle='-', label=u'Témoin(large)')     
    plt.legend()
    plt.plot(ZIs['zi0moy']+ZIs['zi0et'],color='r', linestyle='-', label='Guadeloupe')        
    plt.plot(ZIs['zi1moy']+ZIs['zi1et'],color='b', linestyle='-', label='Martinique')        
    plt.plot(ZIs['zi2moy']+ZIs['zi2et'],color='g', linestyle='-', label='Iles_du_Nord')       
    plt.plot(ZIs['zi3moy']+ZIs['zi3et'],color='y', linestyle='-', label=u'Témoin(large)')    
    
    plt.plot(ZIs['zi0moy']-ZIs['zi0et'],color='r', linestyle='-', label='Guadeloupe')        
    plt.plot(ZIs['zi1moy']-ZIs['zi1et'],color='b', linestyle='-', label='Martinique')        
    plt.plot(ZIs['zi2moy']-ZIs['zi2et'],color='g', linestyle='-', label='Iles_du_Nord')       
    plt.plot(ZIs['zi3moy']-ZIs['zi3et'],color='y', linestyle='-', label=u'Témoin(large)')  
    #plt.plot(arr['zi1et'], label='Ecart-type')
    #plt.axis([0,arr.all['date'],0,1])
    ymin=[0,0,0,0,20,0,0,][counter]
    ymax=[0.15,0.8,0.0006,200,33,0.8,100][counter]    
    
    axes=plt.gca()
    axes.set_ylim([ymin,ymax])
    
    plt.title(fname) 
    
    plt.xlabel('Date')
    #plt.ylabel('[chlor-a] : mg/m3')

    #print 'Guadeloupe, Martinique, Iles de Nord, Large'
    #print 'zi0,         zi1,        zi2,        zi3'
    # ---------- Mise à l'échelle de l'axe des x ---------------
    h=[date[u] for u in range(0,size(date),46)]             ### 8D # 'Pas' de 46 pour tomber sur le même mois (~8jours/365)
    h = [int(round(x/10000000000)) for x in h]    
    p=range(0,size(date),46)                                       # ~13 valeurs, de 2002 à 2015.
    plt.xticks(p,h, fontsize=9, rotation=90)                # re-échelonnage de l'axe des x
    subplots_adjust(bottom=0.15)
    # ----------------------------------------------------------
    #plt.savefig(data_out+sensor+'_'+reso+'_'+variable+'_Graphique.png')
    
    plt.show()
    print savepath+fname+'.png'
    plt.savefig(savepath+fname+'.png',format='png',dpi=300,bbox_inches='tight')
    #raw_input('press to go')
    
    plt.close()
    counter=counter+1
    #print file

