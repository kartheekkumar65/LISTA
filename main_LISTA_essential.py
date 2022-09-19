#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 11:41:45 2021

@author: kartheek

This file implements training and testing setup for LISTA
"""

import torch
import numpy as np
from LISTA_essential import LISTA_train, LISTA_test

def pes(x,x_est):
  d = []
  for i in range(x.shape[1]):
    M = max(np.sum(x[:,i] != 0),np.sum(x_est[:,i] != 0))
    pes_ = (M - np.sum((x[:,i]!=0) * (x_est[:,i]!=0)))/M
    if not np.isnan(pes_):
        d.append(pes_)
    else:
        print(M)
        print('nan is found here')
  return np.mean(d),np.std(d)

def data_gen( D, noise_Std_Dev, k, p, rng):
  print(' noise level is ', noise_Std_Dev)
  m, n = D.shape
  x = rng.normal(0,1,(n,p)) * rng.binomial(1,k,(n,p))
  
  y = D@x 
  noise = rng.normal(0,noise_Std_Dev,y.shape)
  y = y + noise

  return x, y

#%% ------------------ MAIN PROGRAM STARTS --------------------------

input_SNR = 0.01
sparsity = 10
numEpochs = 100
numLayers = 15

seed = 80
print('seed: ', seed)
rng = np.random.RandomState(seed)

m = 70; n = 100;
# create the random matrix
D = rng.normal(0, 1/np.sqrt(m), [m, n])
D /= np.linalg.norm(D,2,axis=0)

numTrain = 100000; numTest = 1000
X_train, Y_train = data_gen( D, input_SNR, sparsity/100, numTrain, rng)
X_test, Y_test = data_gen( D, input_SNR, sparsity/100, numTest, rng)

Y_train = Y_train
Y_test = Y_test

#%%
# retrieve the Signal from Y and D using LISTA using true ground truth

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

learning_rate = 5e-4
import time
start = time.time()

net = LISTA_train(X_train, Y_train, D, numEpochs, numLayers, device, learning_rate)
print(f'time taken is {time.time() - start}')

print('-'*20)
print('Learned threshold is ', net.thr.T)

#%%    
# Sparse Estimation using LISTA (testing phase)
PES_list = []; SNR_mean_list = []
SNR_list = [];
X_out = LISTA_test(net, Y_test, D, device)

N_test = Y_test.shape[1]
for i in range(N_test):
    err = np.linalg.norm(X_out[:, i] - X_test[:, i])
    RSNR = 20*np.log10(np.linalg.norm(X_test[:, i])/err)
    SNR_list.append(RSNR)

SNR_mean_list.append( np.mean(SNR_list))
PES_mean, PES_std = pes(X_test, X_out)

SNR_list_LISTA = np.array(SNR_list)

print('Testing: my LISTA avg SNR is ', np.mean(SNR_list_LISTA))
print('Testing: my LISTA std deviation in SNR is ', np.std(SNR_list_LISTA))
print('Testing: my LISTA peak SNR is ', np.max(SNR_list_LISTA))

print('Testing: my LISTA avg PES is ', PES_mean)
print('Testing: my LISTA std deviation in PES is ', PES_std)
print('-'*40)

# %%
