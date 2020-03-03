import numpy as np
from utility_functions import *


def rnn_cell_forward(xt, a_prev, parameters):
    """
    Implements a single forward step of the RNN-cell"

    """

    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]

    a_next = np.tanh((np.dot(Wax,xt)+np.dot(Waa,a_prev)+ba))
    yt_pred = softmax(np.dot(Wya,a_next)+by)   
    cache = (a_next, a_prev, xt, parameters)
    
    return a_next, yt_pred, cache

def rnn_forward(x, a0, parameters):
    """
    Implement the forward propagation of the recurrent neural network

    """

    caches = []

    n_x, m, T_x = x.shape
    n_y, n_a = parameters["Wya"].shape
    
    a = np.zeros((n_a,m,T_x))
    y_pred = np.zeros((n_y,m,T_x))
    
    a_next = np.copy(a0)

    for t in range(T_x):
        xt = x[:,:,t]
        a_next, yt_pred, cache = rnn_cell_forward(xt,a_next,parameters)
        a[:,:,t] = a_next
        y_pred[:,:,t] = yt_pred
        caches.append(cache)
        
    caches = (caches, x)
    
    return a, y_pred, caches

def lstm_cell_forward(xt, a_prev, c_prev, parameters):
    """
    Implement a single forward step of the LSTM-cell.

    """

    Wf = parameters["Wf"] # forget gate weight
    bf = parameters["bf"]
    Wi = parameters["Wi"] # update gate weight 
    bi = parameters["bi"]
    Wc = parameters["Wc"] # candidate value weight
    bc = parameters["bc"]
    Wo = parameters["Wo"] # output gate weight
    bo = parameters["bo"]
    Wy = parameters["Wy"] # prediction weight
    by = parameters["by"]
    
    n_x, m = xt.shape
    n_y, n_a = Wy.shape

    concat = np.concatenate((a_prev,xt),axis=0)

    ft =sigmoid(np.dot(Wf,concat)+bf)        # forget gate
    it = sigmoid(np.dot(Wi,concat)+bi)        # update gate
    cct = np.tanh(np.dot(Wc,concat)+bc)       # candidate value
    c_next = ft*c_prev+it*cct    # cell state
    ot = sigmoid(np.dot(Wo,concat)+bo)       # output gate
    a_next = ot*np.tanh(c_next)    # hidden state

    yt_pred = softmax(np.dot(Wy,a_next)+by)

    cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters)

    return a_next, c_next, yt_pred, cache

def lstm_forward(x, a0, parameters):
    """
    Implement the forward propagation of the recurrent neural network using an LSTM-cell
    """
    caches = []
    
    Wy = parameters['Wy']
    n_x, m, T_x = x.shape
    n_y, n_a = Wy.shape
    
    a = np.zeros((n_a,m,T_x))
    c = np.zeros((n_a,m,T_x))
    y = np.zeros((n_y,m,T_x))
    

    a_next = np.copy(a0)
    c_next = np.zeros((n_a,m))
    

    for t in range(T_x):
        xt = x[:,:,t]
        a_next, c_next, yt, cache = lstm_cell_forward(xt,a_next,c_next,parameters)
        a[:,:,t] = a_next
        c[:,:,t]  = c_next
        y[:,:,t] = yt
        caches.append(cache)

    caches = (caches, x)

    return a, y, c, caches

def rnn_cell_backward(da_next, cache):
    """
    Implements the backward pass for the RNN-cell (single time-step).

    """

    (a_next, a_prev, xt, parameters) = cache
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]
    
    dz =  (1 - np.square(np.tanh(np.dot(Wax,xt)+np.dot(Waa,a_prev)+ba)))
    dtanh= da_next * dz    
    dxt = np.dot(Wax.T,dtanh)
    dWax = np.dot(dtanh,xt.T)
    da_prev = np.dot(Waa.T,dtanh)
    dWaa = np.dot(dtanh,a_prev.T)
    dba = np.sum(dtanh,axis=1,keepdims=True)

    gradients = {"dxt": dxt, "da_prev": da_prev, "dWax": dWax, "dWaa": dWaa, "dba": dba}
    
    return gradients