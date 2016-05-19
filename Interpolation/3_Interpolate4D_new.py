# -*- coding: utf-8 -*-
"""
Created on Wed May 18 15:01:42 2016

@author: TPhilippon
"""
import numpy as np
import glob
from PIL import Image
from scipy.interpolate import LinearNDInterpolator, griddata
import matplotlib.pyplot as plt

mat1=matrix[0,0,:,:]
mat2=matrix[0,1,:,:]

m=mat1[30:130,:50]
n=mat2[30:130,:50]


x=np.indices(n.shape)[0]
y=np.indices(n.shape)[1]
x=x.flatten()
y=y.flatten()
mm=m.flatten()
nn=n.flatten()

d0=np.zeros(nn.shape)
d10=np.ones(nn.shape)*10
d5=np.ones(nn.shape)*5

xx=np.vstack((x,x)).flatten()
yy=np.vstack((y,y)).flatten()
mn=np.vstack((mm,nn)).flatten()
dd=np.vstack((d0,d10)).flatten()

a=griddata((xx,yy,dd),mn,(x,y,d5),method='linear')

img = np.zeros(n.shape)
img[x,y]= a

plt.imshow(m)
plt.clf()
plt.imshow(n)
plt.clf()
plt.imshow(img)


#____

#Superposer d1 et d2 sur la même array. Addition : 0, 1 et 2.
#On veut un gradient entre les 2 limites. On veut interpoler dans la zone du 1,
#et c'est comme interpoler dans les nuages (NaN).
#On reprend ensuite le travail sur les coordonnées avec zrON et zrNAN. (Ou bien
# ce script là)
# Le nombre de jours doit être modifiable en début de script. Voir pendant combien
# de jours le panache le recouvre une île. 









