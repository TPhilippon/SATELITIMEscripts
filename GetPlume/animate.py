# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 16:06:40 2016

@author: pam
"""

import matplotlib
matplotlib.use('TkAgg') # do this before importing pylab
import matplotlib.pyplot as plt
from PIL import Image
import glob
#import os


path = 'Z:/01_Scripts/PARALLEL/Multiproc_Img/Multiproc_Img/images'
path = 'Y:/SATELITIME/data/ZR/aqua/sst11mic_32d/9km/no_interp'

fig = plt.figure()
ax = fig.add_subplot(111)

def animate():
    filenames=sorted(glob.glob(path+'/mamba_*.png'))
    #print filenames
    im=plt.imshow(Image.open(filenames[0]))
    for filename in filenames[1:]:
        image=Image.open(filename)
        image=image.transpose(Image.FLIP_TOP_BOTTOM)
        im.set_data(image)
       
        fig.canvas.draw() 
        
win = fig.canvas.manager.window
fig.canvas.manager.window.after(100, animate)
plt.show()