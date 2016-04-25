# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 19:33:10 2016

@author: admin
"""
##########
# Utilisation de la "Pool" pour remplacer l'utilisation d'une boucle for
# pour traiter X images contenues dans un dossier.
# L'objetif est de traiter 1 image par processus et d'optimiser le temps de
# traitement.
##########

from multiprocessing import Pool, Array, Process
import sys
import os
import mamba3D as mamba3D
import mamba as mamba
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import time

#outpath = 'Y:/SATELITIME/data/MAMBA/'


 ############################## DONNEES ##############################
    # Liste les images contenues dans le dossier
def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


 ############################## Morphomat ##############################
def tlaferm((name,thresh,outpath)):

    # =============== Load numpy file and create contours then save png =========
#    threshold=25
    # print 'name= ',name
    zr=np.load(name)
    im=Image.fromarray(zr.T)
#    
    xpixels,ypixels = im.size[1],im.size[0]
    dpi,scalefactor = 72,1
    xinch = xpixels * scalefactor / dpi
    yinch = ypixels * scalefactor / dpi
#    
    fig = plt.figure(figsize=(xinch,yinch))
    ax = plt.axes([0, 0, 1, 1], frame_on=False, xticks=[], yticks=[])
    imc=plt.contourf(zr, [0,thresh], colors='black', origin='image') 
##    

    imfile='image_'+os.path.basename(name)+'.png'
    impath=os.path.dirname(name)
    
    plt.savefig(os.path.join(outpath,imfile), dpi=dpi)
    plt.close('all')
#    
    # =================== load png and use mamba then save  ======================================
    im = mamba.imageMb(os.path.join(outpath,imfile))
    im1,im2,im3,im4=mamba.imageMb(im),mamba.imageMb(im),mamba.imageMb(im),mamba.imageMb(im)
    se = mamba.structuringElement([0,1,2,3,4,5,6],grid=mamba.HEXAGONAL) 
    
    mamba.closeHoles(im, im)
    mamba.negate(im, im)
    mamba.closeHoles(im, im)
    mamba.negate(im, im)
#    
    #mamba.opening(im, im1, n=2, se=se)
    ma_file='mamba_'+imfile+'.png'
    im.save(os.path.join(outpath,ma_file))
#    


def data_type():
    outpath = 'Y:/SATELITIME/data/MAMBA/'
    inpath = 'Y:/SATELITIME/data/ZR/'
    os.chdir(inpath)
    dirlist=get_immediate_subdirectories(inpath)
    for i, val in enumerate(dirlist):
        print i,  val

    dirnum=raw_input('What data ? : ') 
    inpath=inpath+dirlist[int(dirnum)]
    outpath=outpath+dirlist[int(dirnum)]
    return inpath,outpath

 #####################################################################
 ############################## Definition des processus #############


# =============================================================





if __name__ == '__main__':
    
    inpath,outpath=data_type()
    print inpath
    print outpath
    #
    data = os.listdir(inpath)
    data=[inpath+'/'+x for x in data]
    nu_items=len(data)
    
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    
    f1=np.load(os.path.join(inpath,data[0]))
    print 'min = ',np.nanmin(f1),'max = ',np.nanmax(f1)
    threshold=0.2 #float(raw_input('What threshold ? : '))
    
    parameter=zip(data,[threshold]*nu_items,[outpath]*nu_items)
    
    start= time.clock()


    pool = Pool() #Pool(processes=12)                      # process per core
    print pool
    #print data
    pool.map(tlaferm,parameter) #,threshold)             # proces data_inputs iterable with pool
    pool.close()
    pool.join()
    
    print time.clock()-start
    
    
