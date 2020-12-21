#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 09:19:36 2018

@author: GangLi
"""


# Fiducial Anutoencoder 11

import keras
from keras import backend as K
import numpy as np

from keras.layers import Input, Dense
from keras.models import Model

from keras.optimizers import RMSprop, Adam

import matplotlib as mpl
#mpl.use('macOsX')

##########################################################
# Generate Data
##########################################################

def DataGenerateFunction(z,mu,m):  
    x = np.transpose(np.tile(mu,(m,1))) + np.matmul(np.diag(np.power(mu,(3/2))),z)
    return x

def model1(m = 10, n = 30):
    mu = np.random.uniform(0, 6, n) 
    z = np.random.normal(0, 1, (n,m))
    x = DataGenerateFunction(z,mu,m)
    return (m, n, x, z, mu)

(m, n, x, z, mu0) = model1(m = 3, n = 100000)

x=x[:,:,np.newaxis]
z=z[:,:,np.newaxis]

r=0.8 # ratio of train and validation

train_X=x[0:int(n*r),:,:]
train_z=z[0:int(n*r),:,:]
valid_X=x[int(n*r):n,:,:]
valid_z=z[int(n*r):n,:,:]

print(train_X.shape[0], 'train samples')
print(valid_X.shape[0], 'test samples')

######################################################
# Define the FAE
######################################################

inChannel = 1

# this is our input placeholder
x_input = Input(shape = (m,  inChannel),name='x')
z_input = Input(shape=(m,  inChannel), name='z')
														 

# "encoded" is the encoded representation of the input
encoded=keras.layers.concatenate([x_input,z_input])
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(128, activation='relu')(encoded)
encoded = Dense(512, activation='relu')(encoded)
encoded = Dense(128, activation='relu')(encoded)
encoded = Dense(512, activation='relu')(encoded)
encoded = Dense(128, activation='relu')(encoded)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(128, activation='relu')(encoded)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)
mu = Dense(1, activation='relu',name='mu_hat')(encoded)

mu1 = keras.layers.AveragePooling1D(pool_size=3, strides=None, padding='valid')(mu)


def dg(ip):
    mu1 = ip[0]
    z2 = ip[1]
    mu2=K.repeat_elements(mu1,3,1) 
    return mu2 + K.pow(mu2,3/2)*z2

decoded = keras.layers.Lambda(dg)([mu1,z_input])
FAE = Model(inputs=[x_input,z_input], outputs=[decoded,mu1])
FAE.compile( optimizer = Adam(),
            loss={'lambda_1': 'mean_squared_error', 
                  'average_pooling1d_1': 'mean_squared_error'},
              loss_weights={'lambda_1': 1, 
                            'average_pooling1d_1': 1})

FAE.summary()

######
# here I fit the true data
batch_size = 250
epochs = 10
FAE_train=FAE.fit({'x': train_X, 'z': train_z},
                  {'lambda_1': train_X, 
                   'average_pooling1d_1': mu0[0:int(n*r),np.newaxis,np.newaxis] },
                  batch_size=batch_size,epochs=epochs,verbose=1,
                  validation_data=({'x': valid_X, 'z': valid_z},
                                   {'lambda_1': valid_X,
                                    'average_pooling1d_1':mu0[int(n*r):n,np.newaxis,np.newaxis] }))


[FAE_pred_X,FAE_pred_mu]=FAE.predict({'x': valid_X, 'z': valid_z})


