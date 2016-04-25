#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Desc : Liste de noms
#
# Auteur : T.P.
# Date : 15/2/15

# ** Téléchargement données '8day' **
#  ** SWF et MODISA possible **

import urllib


day= 1
day2= 0
b= 8
counter=1
path='/home/pressions/SATELITIME/data/chl_8d/'

#for a in range (2002,2016):
for a in range (1997,2011):
#La valeur 'a' s'arrêtera à 2015
    print a
    while day2 < 365:
        day2= day+7
        if day2 > 365:
            day2 = 365
        if a % 4 == 0 and day2 == 365:
            day2 = 366
#Pour que le dernier fichier entre jour 361 et jour 365 soit nommé correctement
#Le dernier fichier pour les années 2004, 2008 et 20012 est en revanche nommé par 361 et 366
            
        url= 'http://oceandata.sci.gsfc.nasa.gov/cgi/getfile/'
        #filen='A'+str(a)+str(format(day,'03'))+str(a)+str(format(day2,'03'))+'.L3m_8D_NSST_4.bz2'  #'.L3m_8D_SST_4.bz2'
        filen='S'+str(a)+str(format(day,'03'))+str(a)+str(format(day2,'03'))+'.L3m_8D_CHL_chlor_a_9km.bz2'  #'.L3m_8D_SST_4.bz2'
                
        #'.L3m_8D_POC_poc_4km.bz2'
        #'.L3m_8D_NSST_4.bz2'
        #'.L3m_8D_CHL_chlor_a_4km.bz2'
        #L3m_8D_CHL_chlor_a_9km.bz2
        print 'Getting', url+filen, path+filen
        #print day
        #print day2
        #print counter
        urllib.urlretrieve(url+filen,path+filen)
        
        counter=counter+1
        day= day+8
        if a == 2015 and day2 == 32:
            day2 = 365
    day= 1
    day2= 0
print 'Fin'

#Fin du script
