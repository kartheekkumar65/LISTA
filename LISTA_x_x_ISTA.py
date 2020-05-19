"""
Author: Kartheek
Date: 29th Jan, 2020 ; 5:22 PM

This file builds and trains the Learned ISTA algorithm
"""
#%%
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import copy

#%%%
def soft_thresh(x, thr):
    # the inputs would be CUDA array elements
    # These elements would be having Auto-grad with them
    mask1 = x > thr
    mask2 = x < -thr
    out = torch.zeros_like(x)
    out += mask1.float() * -thr + mask1.float() * x
    out += mask2.float() * thr  + mask2.float() * x
    return out

#%%
class LISTA(nn.Module):
    def __init__(self, m, n, Dict, numIter, alpha, thr, device, learn_type = 0):
        super(LISTA, self).__init__()

        self._W = nn.Linear(in_features = m, out_features = n, bias=False)
        self._S = nn.Linear(in_features = n, out_features = n, bias=False)
        self.thr = nn.Parameter(torch.tensor(thr))

        self.soft = nn.Softshrink(thr)
        self.numIter = numIter
        self.A = Dict
        self.alpha = alpha
        self.device = device
        
    # custom weights initialization called on network
    def weights_init(self):
        A = self.A
        alpha = self.alpha
        S = torch.from_numpy(np.eye(A.shape[1]) - (1/alpha)*np.matmul(A.T, A))
        S = S.float().to(self.device)
        B = torch.from_numpy((1/alpha)*A.T)
        B = B.float().to(self.device)
        
        self._S.weight = nn.Parameter(S)
        self._W.weight = nn.Parameter(B)


    def forward(self, y):
        
        # x = F.softshrink(self._W(y), self.thr)
        # print('The threshold is:', self.thr)
        # x = self.soft(self._W(y))
        x = soft_thresh(self._W(y), self.thr)

        if self.numIter == 1 :
            return x

        for iter in range(self.numIter):
            # x = soft_thresh(self._W(y), self.thr)
            # x = F.softshrink(self._W(y) + self._S(x), self.thr)
            # x = self.soft(self._W(y) + self._S(x))
            x = soft_thresh(self._W(y) + self._S(x), self.thr)

            # with torch.no_grad():
            #     if iter % 10 == 0:
            #         a = x.detach().data.numpy()
            #         print(iter, ': X is ', a[0,:3])

        return x

# Support metric function
def support_metric(x, x_hat):
    num = np.intersect1d(x.nonzero()[0], x_hat.nonzero()[0]).shape[0]
    denom = max( x.nonzero()[0].shape[0], x_hat.nonzero()[0].shape[0] )
    return num/denom

#%% 
# ISTA functionality for LISTA training




def ista( b, A, l, L, maxit):
    x = np.zeros(A.shape[1])
    err_list = []
    for _ in range(maxit):
        x = soft_thresh(x + np.dot(A.T, b - A.dot(x)) / L, l / L)
        err_list.append(np.linalg.norm(b - A.dot(x)))

    return x, err_list #, pobj, times   

def ISTA_help(X, Y, D):
    # regularization paramter
    _lambda = 0.5 * 15
    # compute the max eigen value of the D'*D
    alpha = (np.linalg.norm(D) ** 2 + 1)
    print('alpha is', alpha)
    print('soft_threshold would be ', _lambda/alpha)

    # retrieve the Signal from Y and D using ISTA
    X_hat = np.zeros(X.shape)
    # computing average reconstruction-SNR
    SNR_list =[]
    support_list = []
    numIter = 3000
    N = Y.shape[1]

    for i in range(N):
        X_hat[:, i], err_list = ista(Y[:, i], D, _lambda , alpha, numIter)
        support_list.append(support_metric(X[:, i], X_hat[:, i]))
        err = np.linalg.norm(X_hat[:, i] - X[:, i])
        RSNR = 20*np.log(1/err)
        SNR_list.append(RSNR)
        

    SNR_list_ista = np.array(SNR_list)
    support_list_ista = np.array(support_list)
    print('Training: my ista avg SNR is ', np.mean(SNR_list_ista))
    print('Training: my ista peak SNR is ', np.max(SNR_list_ista))

    print('Training: my ista avg Support metric is ', np.mean(support_list_ista))
    print('Training: my ista peak Support metric is ', np.max(support_list_ista))

    idx = np.random.randint(0, high = N)
    fig, ax = plt.subplots(1,2)
    ax[0].stem(X_hat[:, idx], use_line_collection = True)
    ax[1].stem(X[:, idx], use_line_collection = True)
    plt.show()

    plt.plot(err_list)
    plt.show()

    return X_hat


#%%
# defining custom dataset
class dataset(Dataset):
    def __init__(self, X, Y):
        super().__init__()
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.Y.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx, :], self.Y[idx, :] # since the input to the datset is tensor


#%%
def my_lista_ISTA_train(X_ISTA, X, Y, D, rng, _lambda, alpha, device = 'cuda:0',  learning_decay = 0.004, numEpochs = 100, learning_rate = 1e-5, learn_type = 0):

    m, n = D.shape

    Train_size = Y.shape[1]
    print('Total dataset size is ', Train_size)
    if Train_size%200 != 0:
        print('Bad Training dataset size')

    batch_size = 400

    numIter = 50
    
    # convert the data into tensors
    Y_t = torch.from_numpy(Y.T)
    Y_t = Y_t.float().to(device)

    # print(Y_t.shape)
    
    D_t = torch.from_numpy(D.T)
    D_t = D_t.float().to(device)

    # we need to use ISTA to get X
    # X_ISTA = ISTA_help(X, Y, D)
    X_t = torch.from_numpy(X_ISTA.T)
    X_t = X_t.float().to(device)

    # dataset_train = dataset(X_t, Y_t)
    dataset_train = dataset(X_t[:-batch_size, :], Y_t[:-batch_size, :])    # we keep the final batch for validation
    print('DataSet size is: ', dataset_train.__len__())
    dataLoader_train = DataLoader(dataset_train, batch_size, shuffle = True)

    # Numpy Random State if rng (passed through arguments)
    net = LISTA(m, n, D, numIter, alpha = alpha, thr = _lambda/(alpha), device = device, learn_type = learn_type)
    net = net.float().to(device)
    net.weights_init()

    S_init = net._S.weight.detach().cpu().data.numpy().copy()
    W_init = net._W.weight.detach().cpu().data.numpy().copy()
    thr_init = net.thr.detach().cpu().data.numpy().copy()

    # build the optimizer and criterion
    
    criterion = nn.MSELoss()
    all_zeros = torch.zeros(batch_size, n).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate, betas = (0.9, 0.999))
    decayRate = 1 - learning_decay
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = decayRate)

    print('Scheduler is on with decay rate of ', decayRate)
    # list of losses at every epoch
    train_loss_list = []
    train_supp_list = []
    valid_loss_list = []
    valid_supp_list = []

    # net = copy.deepcopy(net)
    opt_valid_support = 0
    opt_epoch = 0

    # ------- Training phase --------------
    for epoch in range(numEpochs):
        
        if epoch % 25 == 0 :
            print('Epoch: ', epoch)
        
        tot_loss = 0

        support_metric_train_tot = 0
        support_metric_valid_tot = 0

        for iter, data in enumerate(dataLoader_train):
            # print(net._S.weight.data[:3,:3])
            # print(net._W.weight.data[:3,:3])

            X_ISTA_batch, Y_batch = data
            # clear the gradients
            optimizer.zero_grad()
            X_batch_hat = net(Y_batch.float())  # get the outputs
            
            with torch.no_grad():
                # compute the support metric
                for i in range(batch_size):
                    support_metric_train_tot += (support_metric(X_ISTA_batch[i, :].detach().cpu().numpy(), X_batch_hat[i, :].detach().cpu().numpy()))

            # compute the loss
            loss = criterion(X_batch_hat.float(), X_ISTA_batch.float()) /batch_size
            tot_loss += loss.detach().cpu().data
            loss.backward()     # compute the gradients
            optimizer.step()    # Update the weights

        with torch.no_grad():
            train_loss_list.append(tot_loss.detach().data) 
            train_supp_list.append(support_metric_train_tot / (Train_size - batch_size))

            # validation
            X_valid_hat = net(Y_t[-batch_size:, :])
            loss = criterion(X_valid_hat.float(), X_t[-batch_size:, :].float()) /batch_size
            valid_loss_list.append(loss.data)

            for i in range(batch_size):
                support_metric_valid_tot += (support_metric(X_t[i, :].detach().cpu().numpy(), X_valid_hat[i, :].detach().cpu().numpy()))
            valid_supp_list.append(support_metric_valid_tot/batch_size)

            if support_metric_valid_tot/batch_size > opt_valid_support:
                opt_valid_support = support_metric_valid_tot/batch_size
                opt_net = copy.deepcopy(net)
                opt_epoch = epoch

        # change the learning rate
        scheduler.step()

    with torch.no_grad():
        X_lista = opt_net(Y_t.float())
        if len(Y.shape) <= 1:
            X_lista = X_lista.view(-1)
        X_out = X_lista.cpu().numpy()
        X_out = X_out.T

        S_learned = opt_net._S.weight.detach().cpu().data.numpy().copy()
        W_learned = opt_net._W.weight.detach().cpu().data.numpy().copy()
        thr_learned = opt_net.thr.detach().cpu().data.numpy().copy()

    return opt_net, X_out, train_loss_list, train_supp_list, valid_loss_list, valid_supp_list, S_init, W_init, S_learned, W_learned, thr_init, thr_learned

def my_lista_ISTA_test(net, Y, D, device = 'cuda:0'):

    # convert the data into tensors
    Y_t = torch.from_numpy(Y.T)
    if len(Y.shape) <= 1:
        Y_t = Y_t.view(1, -1)
    Y_t = Y_t.float().to(device)

    # print(Y_t.shape)
    
    D_t = torch.from_numpy(D.T)
    D_t = D_t.float().to(device)

    with torch.no_grad():
        X_lista = net(Y_t.float())
        if len(Y.shape) <= 1:
            X_lista = X_lista.view(-1)
        X_out = X_lista.cpu().numpy()
        X_out = X_out.T


    return X_out