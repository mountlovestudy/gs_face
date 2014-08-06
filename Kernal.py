# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 10:28:57 2014
return the covariance matrix
@author: mountain
"""
import numpy as np
import numpy.matlib as mat
def Kernal(X,theta):
    #the Kernal return the covarience matrix
    #X: the feature data
    #theta:the hyper para, a column vector
    #n_data: the number of the feature in X
    n_data=X.shape[0]
    K=mat.zeros([n_data,n_data])
    for i in range(n_data):
        for j in range(i):
            x1=X[i]
            x2=X[j]
            K[i,j]=theta[0]*np.exp(-0.5*np.dot(np.square(x1-x2),theta[1:-2]))+theta[-2]+(i==j)/theta[-1]
            K[j,i]=K[i,j]
    return K