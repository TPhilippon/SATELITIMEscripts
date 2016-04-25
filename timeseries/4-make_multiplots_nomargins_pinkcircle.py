# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 17:51:46 2015

@author: pam
"""

import os,sys
#import matplotlib.pyplot as plt
#import matplotlib as mpl
#import matplotlib.image as mpimg
#import numpy as np
import time
import Image


format='png'
if os.path.exists('/home/pressions'):
    prefix='/home/'
else:
    prefix='/media/'
origin='upper'
print 'Running...'

left  = None  # the left side of the subplots of the figure
right = None #10.01    # the right side of the subplots of the figure
bottom = None #0.00001 #0.000001   # the bottom of the subplots of the figure
top = None #0.9 #0.00001 #10.01      # the top of the subplots of the figure
wspace = 0.00001   # the amount of width reserved for blank space between subplots
hspace = 0.0000001  # the amount of height reserved for white space between subplots
    
h1,h2=100,1563
v1,v2=60,1070

path1=prefix+'pressions/SATELITIME/data/PNG/aqua/chl_32d/9km/png_caraibe/no_interp/'
path2=prefix+'pressions/SATELITIME/data/PNG/aqua/poc_32d/9km/png_caraibe/no_interp/'
path3=prefix+'pressions/SATELITIME/data/PNG/aqua/sst11mic_32d/9km/png_caraibe/no_interp/'
path4=prefix+'pressions/SATELITIME/data/PNG/aqua/GIOP_adg_32d/9km/png_caraibe/no_interp/'
path5=prefix+'pressions/SATELITIME/data/PNG/aqua/pic_32d/9km/png_caraibe/no_interp/'
path6=prefix+'pressions/SATELITIME/data/PNG/aqua/nsst_8d/9km/png_caraibe/no_interp/'
pathfig=prefix+'pressions/SATELITIME/data/PNG/multiplots/'
pathcropped=prefix+'pressions/SATELITIME/data/PNG/multi-cropped/'

overlayfile=prefix+'pressions/SATELITIME/data/PNG/aqua/overlay_9km_.tif'
blank=prefix+'pressions/SATELITIME/data/PNG/aqua/blank.png'

files1=os.listdir(path1)
files2=os.listdir(path2)
files3=os.listdir(path3)
files4=os.listdir(path4)
files5=os.listdir(path5)
files6=os.listdir(path6)

files1.sort()
files2.sort()
files3.sort()
files4.sort()
files5.sort()
files6.sort()

print len(files1),len(files2),len(files3),len(files4),len(files5),len(files6)

#plt.clf()  

imgol=Image.open(overlayfile)
imgblank=Image.open(blank)


#imgol=imgol.convert("RGBA")
imgol=imgol.crop((h1,v1,h2,v2))
#imgol=imgol[v1:v2,h1:h2]


for fn in range(0,578): #len(files1)):
    print fn
    
    start_time=time.time()
    figname=pathfig+'multi'+str(fn).zfill(4)+'.'+format
    if not os.path.exists(figname):
        #fig=plt.figure()
        imgbl=imgblank
        print 'file number ',fn
           
        img1=Image.open(path1+files1[fn+3])
        img2=Image.open(path2+files2[fn])
        img3=Image.open(path3+files3[fn])
        img4=Image.open(path4+files4[fn])
        img5=Image.open(path5+files5[fn])
        img6=Image.open(path6+files6[fn])
                
        img1=img1.crop((h1,v1,h2,v2))
        img2=img2.crop((h1,v1,h2,v2))
        img3=img3.crop((h1,v1,h2,v2))
        img4=img4.crop((h1,v1,h2,v2))
        img5=img5.crop((h1,v1,h2,v2))
        img6=img6.crop((h1,v1,h2,v2))        
        
        
        anchor=(5,-10)
        img1.paste(imgol,anchor,imgol)
        img2.paste(imgol,anchor,imgol)
        img3.paste(imgol,anchor,imgol)
        img4.paste(imgol,anchor,imgol)
        img5.paste(imgol,anchor,imgol)
        img6.paste(imgol,anchor,imgol)
        
        
        imgbl.paste(img1,(0,0))
        imgbl.paste(img2,(1463,0))
        imgbl.paste(img3,(2926,0))
        imgbl.paste(img4,(0,1010))
        imgbl.paste(img5,(1463,1010))
        imgbl.paste(img6,(2926,1010))
        #im1=Image.blend(img8,imgol,0.5)
        #plt.imshow(im1)
        fn=str(fn)
        imgbl=imgbl.resize((2194,1010),Image.ANTIALIAS)
        imgbl.save(figname)
        #imgbl.show()
        #sys.exit()
        del imgbl
        del img1
        del img2
        del img3
        del img4
        del img5
        del img6

    print("--- %s seconds ---" % (time.time()-start_time)),fn
