#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 22:02:41 2018

@author: allanmartins
"""

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import sys




def model(x, P):
    """
    This is the model to be trainded
    
    This function gives teh "shape" of the function to be optimized
    
    x: Vector with several x values.
        If the vector is a placeholders vector, its computing the TF model.
        If they are numpy arrays, its computing the model output for 
        testing
        
    P: Dictionary with model parameters. 
        If those parameters are variables, its computing the TF model.
        If they are numpy arrays, its computing the model output for 
        testing    
    """

    # y_j = sum_i( exp(x_j - Cx_i)^2/sigmaRbf2 )
    
    M = tf.expand_dims(x,0)-tf.expand_dims(P['Cx'],1)
    arg = tf.squeeze( M*M/P['sigmaRbf2'] )

    y = tf.tensordot(P['W'], tf.exp(-arg), 1)
    
    return y


def organizeData(trainPercentage, filename, col):
    """
    Organize the data for training, testing and presentation 
    
    trainPercentage is the percentage of points used to train (the rest will be
    optionally for validation)
    
    returns a bunch of stuff, but basically input and output for training, testing
    and the complete set (t,f)
    
    """
    
    data = np.loadtxt(filename)
    
    y = data[:,6] - np.mean(data[:,col])
    
    
    # number of samples
    NTrain = int(data.shape[0]*trainPercentage)
    NTest = data.shape[0]-NTrain
    
    t = np.linspace(0, NTrain+NTest-1, NTrain+NTest)
    
    idxTrain = np.random.choice(NTrain+NTest, NTrain)
    idxTest = np.random.choice(NTrain+NTest, NTest)
    (t_train, y_train) = (np.array([t[idxTrain]]), np.array([y[idxTrain]]))
    (t_test, y_test) = (np.array([t[idxTest]]), np.array([y[idxTest]]))

    return ((t, y), (t_train, y_train), (t_test, y_test), (NTrain, NTest)) 





# main

if len(sys.argv)!=4:
    print("Usage: rby.py inputfile colum_number outputfile")
    exit()



# assemble data
(t, y), (t_train, y_train), (t_test, y_test), (NTrain, NTest) = organizeData(1.0, sys.argv[1], int(sys.argv[2]))



# training parameters
learning_rate = 1.5
momentum = 0.1
sigma2 = 1




# sizes and dimensions
inputDimension = t_train.shape[0]
outputDimension = y_train.shape[0]

batch_size = int(t_train.shape[1]/10)



# feature space dimension
nW = 4*19



# initial model parameters
W = 5*np.cos(np.linspace(1,nW,nW)*np.pi/2)
Cx = np.linspace(0,np.max(t),nW)
sigmaRbf2 = 10000


# placeholders for training
tPlace = tf.placeholder(tf.float32, shape=(inputDimension, batch_size))
yPlace = tf.placeholder(tf.float32, shape=(outputDimension, batch_size))


# model varaibles
tfW = tf.Variable(W,dtype=tf.float32)
tfCx = tf.Variable(Cx,dtype=tf.float32)
tfSigmaRbf2 = tf.Variable(sigmaRbf2,dtype=tf.float32)


# model output
yHat = model(tPlace, {'W':tfW, 'Cx':tfCx, 'sigmaRbf2':tfSigmaRbf2})






# loss function (to be optimized)
loss = -tf.reduce_mean( tf.exp( -tf.square(yPlace - yHat)/sigma2 ) )  
#loss = tf.reduce_mean( tf.square(yPlace - yHat) )   





# optimizer: just the algorithm, not to be runned directly
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum = momentum)

# step: when runned, updates the variables associated with loss acording to 
# the method in optmizer
step = optimizer.minimize(loss)

# global initializer. Must be called before training
init = tf.global_variables_initializer()

print("training...")

_minLoss = 100000000.0
with tf.Session() as sess:
    
    # initialize
    sess.run(init)

    # train for this much time...
    for i in range(50000):
        
        # random indices to form a batch for traning
        index = np.random.choice(NTrain, batch_size, replace=False) 
        
        # steps the optimization algorithm and returns the current loss
        _loss, _ = sess.run([loss, step], {tPlace:t_train[:, index], yPlace:y_train[:, index]})
        
        
        # display improvements 
        if _loss < _minLoss:
            _minLoss = _loss
            
            # keep the good results
            WOut, CxOut, sigmaRbf2Out  = sess.run([tfW, tfCx, tfSigmaRbf2])
        
            yHat = sess.run( model(t.astype(np.float32),{'W':WOut, 'Cx':CxOut, 'sigmaRbf2':sigmaRbf2Out}) )
            
        
            print(_minLoss)

plt.plot(t,yHat,t,y)
plt.show()

print('\n\n')


fid = open(sys.argv[3], 'wt')

for yh in yHat:
    fid.write('%.4f '%yh)

fid.close()