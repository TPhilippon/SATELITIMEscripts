# -*- coding: cp1252 -*-
import matplotlib
matplotlib.use('TkAgg') # do this before importing pylab
import matplotlib.pyplot as plt
import Image
import glob
import os

if os.path.exists('/home/pressions'):
    prefix='/home/'
else:
    prefix='/media/'
    
pathfig=prefix+'pressions/SATELITIME/data/PNG/multiplots/morph'
#print os.getcwd()


fig = plt.figure()
ax = fig.add_subplot(111)

def animate():
    filenames=sorted(glob.glob(pathfig+'/*.png'))
    print filenames
    im=plt.imshow(Image.open(filenames[0]))
    for filename in filenames[1:]:
        image=Image.open(filename)
        #image=image.transpose(Image.FLIP_TOP_BOTTOM)
        im.set_data(image)
       
        fig.canvas.draw() 

win = fig.canvas.manager.window
fig.canvas.manager.window.after(0, animate)
plt.show()

