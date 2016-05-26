# -*- coding: utf-8 -*-
"""
Created on Thu May 26 17:12:05 2016

@author: terencephilippon
"""

#mattest=np.copy(matdXb)
#
##mattest[mattest==1]=0
#mattest[mattest<0.4]
#
#mattest[mattest>0.4]=0

mattest=np.copy(matdXb)

mattest[mattest<0.4]=-1

mattest[mattest>=0.4]=0

mattest[mattest==-1]=1
#______

mattest2=np.copy(matdXa)

mattest2[mattest2>0.4]=1

mattest2[mattest2<=0.4]=0


#_____

A = mattest+mattest2
A[A==2]=1