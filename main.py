#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 17:00:31 2020

@author: kartheek

This file has testing features for sparse signal estimation
"""

#%%
# Import stuff


import numpy as np
from ISTA import my_ista, ista
import matplotlib.pyplot as plt
import LISTA_x_x_ISTA
import time 
import helpers_LISTA 
import importlib

#%%
# defninig the dimensions etc..
# dimensions of the sparse signal
n = 256 # 40

# dimension of the compressed signal
m = 150 # 20

# sparsity
k = 25 # 5

# number of examples for training
N_train = 12000

# number of examples we would be testing
N_test = 2000

# A state of randomness
# seed = np.random.randint(low=1, high=100)
seed = 96
print('seed: ', seed)
rng = np.random.RandomState(seed)

# create the random matrix
D = rng.randn(m, n)

# for i in range(n):
#     D[:, i] /= np.linalg.norm(D[:, i])

# noise observations 
SNR = 20
noise_std = np.power(10, -(SNR/20))

#%%
# Data Generation
# generate the training set
X_train = np.zeros([n, N_train])
Y_train = np.zeros((m, N_train))
for i in range(N_train):
    sparse_locs = rng.randint(0, n, [k,])

    X_train[sparse_locs, i ] = 2*rng.randn(k)
    X_train[:, i] /= np.linalg.norm(X_train[:, i])
    Y_train[:, i] = np.matmul(D, X_train[:, i]) + noise_std*rng.randn(m)

# generate the test set
X_test = np.zeros([n, N_test])
Y_test = np.zeros((m, N_test))
for i in range(N_test):
    sparse_locs = rng.randint(0, n, [k,])

    X_test[sparse_locs, i ] = 2*rng.randn(k)
    X_test[:, i] /= np.linalg.norm(X_test[:, i])
    Y_test[:, i] = np.matmul(D, X_test[:, i]) + noise_std*rng.randn(m)

# compute the max eigen value of the D'*D
alpha = (np.linalg.norm(D) ** 2 + 1)

#%%
# Support metric function
def support_metric(x, x_hat):
    num = np.intersect1d(x.nonzero()[0], x_hat.nonzero()[0]).shape[0]
    denom = max( x.nonzero()[0].shape[0], x_hat.nonzero()[0].shape[0] )
    return num/denom

#%%
# -------------------- Version 2 ------------------------

importlib.reload(helpers_LISTA)
importlib.reload(LISTA_x_x_ISTA)

# retrieve the Signal from Y and D using LISTA using true ground truth
lista_lambda = 0.1 * 850
version = 0 # Regular LISTA implementation with normal cost function

learning_rate = 10e-5
numEpochs = 100
learning_decay = 0.008
print('LISTA LAmbda is ', lista_lambda)

net = helpers_LISTA.LISTA_train_wrap(X_train, X_train, Y_train, D, lista_lambda, alpha, rng, version, learning_decay , numEpochs, learning_rate)

print('Epochs is ', numEpochs)
print('learning decay is ', learning_decay)
print('LISTA LAmbda is ', lista_lambda)
print('learning_rate is ', learning_rate)

# Sparse Estimation using LISTA (testing phase)
helpers_LISTA.LISTA_test_wrap(net, X_test, Y_test, D)

