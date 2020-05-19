""" 
Author: Kartheek Reddy  
Date: 2020-02-20 11:36:35 
Goal: This file contains the wrapped functionalities of ISTA and various versions of Learned ISTA
"""

#%%
import numpy as np
from ISTA import my_ista, ista
import matplotlib.pyplot as plt
import LISTA_x_x_ISTA
import LISTA_y_Ax_x1
import time 
import importlib

#%%
# Support metric function
def support_metric(x, x_hat):
    num = np.intersect1d(x.nonzero()[0], x_hat.nonzero()[0]).shape[0]
    denom = max( x.nonzero()[0].shape[0], x_hat.nonzero()[0].shape[0] )
    return num/denom

def ISTA_wrap(X, Y, D, _lambda , alpha):
    # retrieve the Signal from Y and D using ISTA
    X_hat = np.zeros(X.shape)
    # computing average reconstruction-SNR
    SNR_list =[]
    support_list = []
    numIter = 2000
    N = Y.shape[1]

    start = time.time()
    for i in range(N):
        X_hat[:, i], err_list = ista(Y[:, i], D, _lambda , alpha, numIter)
    end = time.time()
    time_ista = end - start

    for i in range(N):
        support_list.append(support_metric(X[:, i], X_hat[:, i]))
        err = np.linalg.norm(X_hat[:, i] - X[:, i])
        RSNR = 20*np.log(1/err)
        SNR_list.append(RSNR)
        

    SNR_list_ista = np.array(SNR_list)
    support_list_ista = np.array(support_list)
    print('Training: time taken {}'.format(time_ista) )
    print('Training: my ista avg SNR is ', np.mean(SNR_list_ista))
    print('Training: my ista peak SNR is ', np.max(SNR_list_ista))

    print('Training: my ista avg Support metric is ', np.mean(support_list_ista))
    print('Training: my ista peak Support metric is ', np.max(support_list_ista))

    fig, ax = plt.subplots(1,2)
    ax[0].stem(X_hat[:, 0], use_line_collection = True)
    ax[1].stem(X[:, 0], use_line_collection = True)
    plt.show()

    plt.plot(err_list)
    plt.show()

    print('-'*40)

    return X_hat

def LISTA_train_wrap(X_ISTA, X_train, Y_train, D, lista_lambda, alpha, rng, version,  learning_decay = 0.004, numEpochs = 100, learn_type = 0, learning_rate = 1e-5):
    # computing average reconstruction-SNR
    SNR_list = []
    support_list = []
    N_train = Y_train.shape[1]
    start = time.time()
    

    # get the learned network
    if version == 0 :
        net, X_train_hat, train_loss_list, train_supp_list, valid_loss_list, valid_supp_list, S_init, W_init, S_learned, W_learned, thr_init, thr_learned = LISTA_x_x_ISTA.my_lista_ISTA_train( X_ISTA, X_train, Y_train, D, rng, lista_lambda, alpha, device = 'cuda:0', learning_decay = 0.004, numEpochs = numEpochs, learn_type = 0, learning_rate = learning_rate)

    time_lista_train = end - start

    for i in range(N_train- 200):
        support_list.append(support_metric(X_train[:, i], X_train_hat[:, i]))
        err = np.linalg.norm(X_train_hat[:, i] - X_train[:, i])
        RSNR = 20*np.log(1/err)
        SNR_list.append(RSNR)


        
    SNR_list_lista = np.array(SNR_list)
    support_list_lista = np.array(support_list)
    print('Training: time taken {}'.format(time_lista_train) )
    print('Traning: my lista avg SNR is ', np.mean(SNR_list_lista))
    print('Traning: my lista peak SNR is ', np.max(SNR_list_lista))

    print('Traning: my lista avg Support metric is ', np.mean(support_list_lista))
    print('Traning: my lista peak Support metric is ', np.max(support_list_lista))

    print('-'*20)


    SNR_list = []
    support_list = []
    for i in range(N_train - 200, N_train):
        support_list.append(support_metric(X_train[:, i], X_train_hat[:, i]))
        err = np.linalg.norm(X_train_hat[:, i] - X_train[:, i])
        RSNR = 20*np.log(1/err)
        SNR_list.append(RSNR)

    SNR_list_lista = np.array(SNR_list)
    support_list_lista = np.array(support_list)
    print('Validation: my lista avg SNR is ', np.mean(SNR_list_lista))
    print('Validation: my lista peak SNR is ', np.max(SNR_list_lista))

    print('Validation: my lista avg Support metric is ', np.mean(support_list_lista))
    print('Validation: my lista peak Support metric is ', np.max(support_list_lista))

    idx = rng.randint(0, high = N_train)
    fig, ax = plt.subplots(1,2)
    ax[0].stem(X_train_hat[:, idx], use_line_collection = True)
    ax[1].stem(X_train[:, idx], use_line_collection = True)
    plt.title('reconstructed vs original')
    plt.savefig('plots//stem_'+ str(learning_decay)+ '_'+ str(numEpochs)+ '_'+ str(learning_rate)+ '_'+ str(lista_lambda)+'.png' )

    print('Decay of error Traning and Validation')
    fig1, ax1 = plt.subplots(1,2)
    ax1[0].plot(train_loss_list)
    ax1[1].plot(valid_loss_list)
    plt.title('training loss and validation loss')
    plt.savefig('plots//err_'+ str(learning_decay)+ '_'+  str(numEpochs)+ '_'+str(learning_rate)+ '_'+ str(lista_lambda)+'.png' )

    print('Training Support and Validation Supoort')
    fig2, ax2 = plt.subplots(1,2)
    ax2[0].plot(train_supp_list)
    ax2[1].plot(valid_supp_list)
    plt.title('training support and validation support')
    plt.savefig('plots//supp_'+ str(learning_decay)+ '_'+  str(numEpochs)+ '_'+str(learning_rate)+ '_'+ str(lista_lambda)+'.png' )

    print('-'*40)

    return net

def LISTA_test_wrap(net, X_test, Y_test, D):
    # retrieve the Signal from Y and D using LISTA
    m, N_test = Y_test.shape
    n = D.shape[1]
    X_test_hat = np.zeros([n, N_test])

    # computing average reconstruction-SNR
    SNR_list = []
    support_list = []

    start = time.time()
    X_test_hat = LISTA_x_x_ISTA.my_lista_ISTA_test( net, Y_test, D)
    end = time.time()
    time_lista_test = end - start

    for i in range(N_test):
        support_list.append(support_metric(X_test[:, i], X_test_hat[:, i]))
        err = np.linalg.norm(X_test_hat[:, i] - X_test[:, i])
        RSNR = 20*np.log(1/err)
        SNR_list.append(RSNR)

    SNR_list_lista = np.array(SNR_list)
    support_list_lista = np.array(support_list)
    print('Testing: time taken {}'.format(time_lista_test) )
    print('Testing: my lista avg SNR is ', np.mean(SNR_list_lista))
    print('Testing: my lista peak SNR is ', np.max(SNR_list_lista))

    print('Testing: my lista avg Support metric is ', np.mean(support_list_lista))
    print('Testing: my lista peak Support metric is ', np.max(support_list_lista))
    print('-'*40)

    # idx = np.random.randint(0, high = N_test)
    # fig, ax = plt.subplots(1,2)
    # ax[0].stem(X_test_hat[:, idx])
    # ax[1].stem(X_test[:, idx])
    # plt.show()