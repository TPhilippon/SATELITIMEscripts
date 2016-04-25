#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Desc : Liste de noms
#
# Auteur : Pierre-Alain Mannoni & Térence Philippon
# Email : Pierre-Alain.MANNONI@unice.fr & terence.philippon@sfr.fr
# Date : 08/7/15

##### ** Script en projet pour MODISA / SWF en 8 et 32 D (L3m) **
#             ** Changer les variables par leur indice **
#       ** Activer / Désactiver les blocs test correspondants **
#################################################################################
# Ce script permet de télécharger des données du site oceandata.sci.gsfc.nasa.gov
# en choisissant certaines des variables disponibles. Il est important de vérifier 
# que tout le jeu de donnée s'est correctement téléchargé (il existe de nombreuses
# exceptions et tests -> donc beaucoup de possibilités d'erreurs). 
# Actier / désactiver les tests correspondants aux variables choisies (satellite, 
# résolution...). Et les modifier si besoin. Vérifier les fichiers après le DL.
# ===============================================================================
# ============= Imports =========================================================

import urllib
import os,sys

# ===============================================================================
# ============== Choisir les variables ==========================================

timer = 1                                   #--> 8 D (0) ou 32 D (1)
varnum = 6                                  #--> 0,1,2,3,4 pour 8D ; 5,6,7,8,9 pour 32D || ordre : Chl,sst,poc,nsst,pic
sat = 1                                     #--> 0 pour SWF et 1 pour MODIS
reso= ['4km', '9km'][1]                     # --> Web & local path >> Verifier le nom des fichiers ('9' ou '9km') !
resofname=['4', '9km'][1]
t1, t2 = [(1997, 2010),(2002,2015)][sat]    # Pour boucle for (t1 et t2+1) pour SWF et MODISA
laps =[7,31][timer]                         # 7 ou 31 jours de décalage (8D ou 32D)
day= 1                                      # Borne jour 1
day2= [8,32][timer]                         # Borne jour 2 (pour 8D et 32D)
a2= int()                                   # Borne Annee 2 ('a' est définie dans la boucle)

sat_path= ['swf','aqua'][sat]                                                               # --> local directory
sat_name= ['S', 'A'][sat]                                                                   # --> Pour le nom du fichier
time_laps= ['8Day', 'Rolling_32_Day'][timer]                                                # --> Web path
var_path= ['chl_8d','sst11mic_8d','poc_8d','nsst_8d', 'pic_8d',
      'chl_32d','sst11mic_32d','poc_32d','nsst_32d', 'pic_32d','GIOP_adg_32d'][varnum]                     # --> local directory                     
filler= ['8D_CHL_chlor_a_','8D_SST_sst_','8D_POC_poc_','8D_NSST_', '8D_PIC_pic_',
         'R32_CHL_chlor_a_','R32_SST_','R32_POC_poc_','R32_NSST_','R32_PIC_pic_','R32_IOP_adg_443_giop_'][varnum]   # --> Web filename

print 'resolution -->', reso,'variable -->', var_path

if os.path.exists('/home/pressions'):
    prefix='/home/'
else:
    prefix='/media/'
origin='upper'
print 'Prefix...', prefix


path=prefix+'pressions/SATELITIME/data/FULL/'             # |
path = path+var_path+'/'+sat_path+'/'+reso+'/'           # --> save path (local)
url= 'http://oceandata.sci.gsfc.nasa.gov/cgi/getfile/'
bisextile=[1980,1984,1988,1992,1996,2000,2004,2008,2012,2016]

# ===============================================================================
# =============== Boucle de téléchargement ======================================

for a in range (t1,t2+1):                   # Boucle de borne1 à borne2+1 (MODISA & SWF)
    print a
    a2 = a
    # --------------------------
    if a == 2002:                           # Test MODISA
        day = 185
        day2 = day+laps
    else:
        day =1
        day2 =day+laps
    # ---------------------------
#    if a == 1997:                          # Test SWF
#        day = 217
#        day2 = day+laps
#    else:
#        day =1
#        day2 = day+laps
    # ---------------------------
    while day <= 361:                       # 361 valable pour SWFS et MODISA (normalement).
        # ----------------ls--- 
        if a in bisextile and day2 > 365:                      # Test MODISA 8D
            day2 = 366
        if not a in bisextile and day2 > 365:
            day2=365
            
        #else :
            #if a % 4 == 0 and day == 365:
         #   day2 = 366
        # --------------------
#        if day == 337:                      # Test 32D (fichiers sur 2 années)
#            day2 = 3
#            a2=a+1
#        # --------------------
#        if a % 4 == 0 and day == 337:       # Test 32D (pour années % par 4)
#            day2 = 2
#            a2=a+1

        
        ##### Générateur du nom de fichier #####
        filen=sat_name+str(a)+str(format(day,'03'))+str(a2)+str(format(day2,'03'))+'.L3m_'+filler+resofname+'.bzip'  
        #'.L3m_8D_POC_poc_4km.bz2'
        #'.L3m_8D_NSST_4.bz2'
        #'.L3m_8D_CHL_chlor_a_4km.bz2'
        #L3m_8D_CHL_chlor_a_9km.bz2
        print filen
# ========================== Télégarchement ===============================
        if not os.path.exists(path+filen[:-7]):  #[:-14]
            print 'Getting', url+filen, '.........', path+filen
            urllib.urlretrieve(url+filen,path+filen)           # -----> Téléchargement
            #sys.exit()

# =========================================================================
        day= day+8
        day2= day2+8
        # ---------------------------------
        if a == 2015 and day2 > 176:        # --> MODISA 8 et 32 Stopeur
            day = 9999
        # ---------------------------------
#        if a == 2010 and day > 345:        # -- > SWF 8 et 32 Stopeur
#            day = 9999
        # ---------------------------------
    day= 1
    day2= day+laps
print 'Fin'

#Fin du script
