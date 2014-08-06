# -*- coding: utf-8 -*-
"""
Created on Wed May 28 22:08:26 2014
GS_Face:
Construct GaussianFace Model
input:
     Xt:    the target domain data
     Xs:    the source domain data xs, may None
     dlta:  in Eq.(13), a global scaling of the prior
     beta:  in Eq.(19), a para balance the relative importance
output:
    the hyper parameter theta

@author: mountain
"""
import theano
import theano.tensor as T
import numpy as np

class GsFace(object):
    
    def __init__(self,X,n_in,n_out):
        """ Initialize the parameters of the GS-FACE-MODEL
        :type X: theano.tensor.TensorType
        :param X: symbolic variable that describes the input of the
        architecture
        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie
        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie
        """
        #the hyper parameter
        self.theta=theano.shared(value=np.zeros((n_out+3,1),dtype=theano.config.floatX),name='theta',borrow=True) 
        self.n_in=n_in
        self.n_out=n_out
        self.K=self.K_mtr(X)
            
    def K_mtr(self,X):
        #return the cov matrix
        K=theano.shared(value=np.zeros([self.n_in,self.n_in],dtype=theano.config.floatX))
        K_new=T.matrix('K_new')
        for i in xrange(self.n_in):
            for j in xrange(self.n_in):
                x1=X[i]
                x2=X[j]
                K_new=T.set_subtensor(K[i,j],self.theta[0]*T.exp((-0.5*T.dot((x1-x2)*2,self.theta[1:-2])))+self.theta[-2]+T.eq(i,j)/self.theta[-1])
                K=K_new
        return K 
                    
        
        
        
    
    def P_prior():
        
        
        pz=0
        return Pz
    
    def P_latent():
        
        pz=0
        return pz
    
    def P_theta():
        
        p_theta=0
        return p_theta
    
    def P_posterior():
    
        p_z_x=0
        return p_z_x
    
    
        
        
        
    
    
    def negative_log_likelyhood(self,input,beta):
        """Return GaussianFace model, learning the gaussian face model amounts to 
            min the negative_log_likelyhood
        """
        #return T.log()
        return 0

x=T.matrix('x')
gs=GsFace(X=x,n_in=2,n_out=2)
y=gs.K
f=T.function(inputs=[x],outputs=y)  
f(np.array([[3,4],[2,3]]))                                              
    
    

