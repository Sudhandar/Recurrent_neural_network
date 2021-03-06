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

def rnn_backward(da, caches):
    """
    Implement the backward pass for a RNN over an entire sequence of input data.

    """
    (caches, x) = caches
    (a1, a0, x1, parameters) = caches[0]
    
    n_a, m, T_x = da.shape
    n_x, m = x1.shape
    
    dx = np.zeros((n_x,m,T_x))
    dWax = np.zeros((n_a,n_x))
    dWaa = np.zeros((n_a,n_a))
    dba = np.zeros((n_a,1))
    da0 = np.zeros((n_a,m))
    da_prevt = np.zeros((n_a,1))
    
    for t in reversed(range(T_x)):
        gradients = rnn_cell_backward(da[:,:,t]+da_prevt,caches[t])
        dxt, da_prevt, dWaxt, dWaat, dbat = gradients["dxt"], gradients["da_prev"], gradients["dWax"], gradients["dWaa"], gradients["dba"]
        dx[:, :, t] = dx[:,:,t]+dxt
        dWax += dWaxt
        dWaa += dWaat
        dba += dbat
        
    da0 = da_prevt

    gradients = {"dx": dx, "da0": da0, "dWax": dWax, "dWaa": dWaa,"dba": dba}
    
    return gradients

def lstm_cell_backward(da_next, dc_next, cache):
    """
    Implement the backward pass for the LSTM-cell (single time-step).

    """

    (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters) = cache
    

    n_x, m = xt.shape
    n_a, m = a_next.shape
    
    dot =da_next*np.tanh(c_next)*ot*(1-ot)
    dcct = (dc_next*it+ot*(1-np.square(np.tanh(c_next)))*it*da_next)*(1-np.square(cct))
    dit = (dc_next*cct+ot*(1-np.square(np.tanh(c_next)))*cct*da_next)*it*(1-it)
    dft = (dc_next*c_prev+ot*(1-np.square(np.tanh(c_next)))*c_prev*da_next)*ft*(1-ft)
    

    concat=np.concatenate((a_prev,xt),axis=0)
    dWf = np.dot(dft,concat.T)
    dWi = np.dot(dit,concat.T)
    dWc = np.dot(dcct,concat.T)
    dWo = np.dot(dot,concat.T)
    dbf = np.sum(dft,axis=1,keepdims=True)
    dbi = np.sum(dit,axis=1,keepdims=True)
    dbc = np.sum(dcct,axis=1,keepdims=True)
    dbo = np.sum(dot,axis=1,keepdims=True)

    da_prev = np.dot(parameters['Wf'][:,:n_a].T,dWf)+np.dot(parameters['Wi'][:,:n_a].T,dWi)+np.dot(parameters['Wc'][:,:n_a].T,dWc)+np.dot(parameters['Wo'][:,:n_a].T,dWo)
    dc_prev = dc_next*ft + ot *(1-np.square(np.tanh(c_next)))*ft*da_next
    dxt = np.dot(parameters['Wf'][:,n_a:].T,dWf)+np.dot(parameters['Wi'][:,n_a:].T,dWi)+np.dot(parameters['Wc'][:,n_a:].T,dWc)+np.dot(parameters['Wo'][:,n_a:].T,dWo)

    gradients = {"dxt": dxt, "da_prev": da_prev, "dc_prev": dc_prev, "dWf": dWf,"dbf": dbf, "dWi": dWi,"dbi": dbi,
                "dWc": dWc,"dbc": dbc, "dWo": dWo,"dbo": dbo}

    return gradients