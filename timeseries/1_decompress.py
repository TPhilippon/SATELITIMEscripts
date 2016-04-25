#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
##### ** Script non fonctionnel ** #####

import sys
import os
import bz2
from bz2 import decompress
print 'start'

varg=['sst11mic_8d','poc_8d', 'nsst_8d','chl_8d']
varg=varg[3]  # Choisir la variable géochimique


prepath='/home/pressions/SATELITIME/data/'+varg+'/seawifs/'

dirpath = varg
dirpathout = varg+'/hdf_swf'
print dirpathout
for(dirpath,dirnames,files)in os.walk(prepath):
        print dirpath
        files.sort()
        for filename in files:
                filepath = os.path.join(prepath, filename)
                newfilepath = os.path.join(prepath,filename + '.hdf')
                print filename
                if not os.path.isfile(newfilepath):
                        print 'Decompressing',  filename, 
                        with open(newfilepath, 'wb') as new_file, open(filepath, 'rb') as file:
                                decompressor = bz2.BZ2Decompressor()
                                for data in iter(lambda : file.read(100 * 1024), b''):
                                        new_file.write(decompressor.decompress(data))

print 'end'