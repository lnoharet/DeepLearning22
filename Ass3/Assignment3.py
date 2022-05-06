__author__ = 'Léo Noharet'
__email__ = 'lnoharet@kth.se'
__date__ = '5/5/2022'

import copy
import datetime
import math
import random
from prettytable import PrettyTable
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import time
import scipy
from scipy.io import *
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt

class NeuralNetwork():

    def __init__(
                 self, n=None, d=None, k=None, K=None, layer_dims=None, 
                 eta_min=1e-5, eta_max=1e-1, 
                 n_step=None, 
                 val_data_size=None, param_init_type = "He", 
                 cycles = 2, 
                 batch_size = 100, 
                 lambda_ = 0.005,
                 alpha = 0.9,
                 sigma = None,
                 lmd_search = False,
                 rand_batch = False,
                 gradient_test = False,
                 plotting = False,
                 batch_norm = False,
                 data_path = '../Datasets/cifar-10-batches-mat/'
                 ):

        self.data_path = data_path
        
        self.net_options = {'lmd_search' : lmd_search, 'rand_batch' : rand_batch, 'gradient_test' : gradient_test, 'plotting' : plotting, 'batch_norm': batch_norm}
        self.net_params = {'W': None, 'b': None, 'gammas': None, 'betas': None}
        self.net_grads  = {'W': None, 'b': None, 'gammas': None, 'betas': None}

        self.mean_s_av = [None] * k
        self.var_s_av = [None] * k
        self.is_s_averages_init = False

        self.net_hyper_params = {
                                 'cycles': cycles, 
                                 'batch_size' : batch_size,  
                                 'lambda_' : lambda_,
                                 'eta_min' : eta_min,
                                 'eta_max' : eta_max,
                                 'n_step' : n_step,
                                 'param_init_type': param_init_type,
                                 'val_data_size' : val_data_size,
                                 'alpha' : alpha
                                 }
        
        self.train_X, self.train_Y, self.train_y = None, None, None
        self.val_X,   self.val_Y,   self.val_y   = None, None, None
        self.test_X,  self.test_y                = None, None

        self.n = n # number of images
        self.d = d # dimension of images
        self.k = k # number of layers
        self.K = K # number of classes
        self.layer_dims = layer_dims # The number of neurons in each hidden layer

        if lmd_search:

            self.lambda_srch_res = None
            self.load_all_data(val_data_size)
            # preprocess data by normalizing
            self.pre_process()
            # set n, d, n_step, K, layer_dims after loading data
            self.set_net_dimensions()
        else:
        # load data
            if val_data_size is None: 
                self.load_some_data()
            else: 
                self.load_all_data(val_data_size)
            # preprocess data by normalizing
            self.pre_process()
            # set n, d, n_step, K, layer_dims after loading data
            self.set_net_dimensions()
            #init parameters W, b, gamma, and betas
            self.init_parameters(param_init_type, sigma)

    

    def set_hyper_params(self, **kwargs):
        for key, value in kwargs.items():
            if key not in self.net_hyper_params:
                raise ValueError('Specified parameter ' + key + ' not a hyperparameter of NN. \n Hyperparameters are: ' + ', '.join(key for key in self.net_hyper_params.keys()))
            else:
                self.net_hyper_params[key] = value

    def set_options(self, **kwargs): 
        for key, value in kwargs.items():
            if key not in self.net_options:
                raise ValueError('Specified parameter ' + key + ' not a network option. Options available: ' + ', '.join(key for key in self.net_options.keys()))
            else:
                self.net_options[key] = value

    def load_all_data(self, val):
        """ Load all except val available data for training datset and val for validation dataset"""
        # val = 5000
        training1 = self.separate_data('data_batch_1.mat', 'training_batch1.mat')
        training2 = self.separate_data('data_batch_2.mat', 'training_batch2.mat')
        training3 = self.separate_data('data_batch_3.mat', 'training_batch3.mat')
        training4 = self.separate_data('data_batch_4.mat', 'training_batch4.mat')
        training5 = self.separate_data('data_batch_5.mat', 'training_batch5.mat')
        test_batch = self.separate_data('test_batch.mat', 'testing_batch.mat')

        train_X = np.concatenate(( training1['data'], training2['data'], training3['data'], training4['data'], training5['data']), axis=1)
        train_Y = np.concatenate(( training1['onehot'], training2['onehot'], training3['onehot'], training4['onehot'], training5['onehot']), axis=1)
        train_y = np.concatenate(( training1['labels'], training2['labels'], training3['labels'], training4['labels'], training5['labels']), axis=0)
        
        self.val_X = train_X[:, -val:]
        self.val_Y = train_Y[:, -val:]
        self.val_y = train_y[-val:, :]
        self.train_X = train_X[:, :-val]
        self.train_Y = train_Y[:, :-val]
        self.train_y = train_y[:-val, :]
        self.test_X = test_batch['data']
        self.test_y = test_batch['labels']    
        return

    def load_some_data(self):
        """ Load only a subset of available data"""
        training_batch    = self.separate_data('data_batch_1.mat', 'training_batch.mat')
        validation_batch  = self.separate_data('data_batch_2.mat', 'validation_batch.mat')
        test_batch        = self.separate_data('test_batch.mat', 'testing_batch.mat')

        self.train_X = np.matrix(training_batch['data'])
        self.train_Y = np.matrix(training_batch['onehot'])
        self.train_y = np.matrix(training_batch['labels'])

        self.val_X = np.matrix(validation_batch['data'])
        self.val_Y = np.matrix(validation_batch['onehot'])
        self.val_y = np.matrix(validation_batch['labels'])

        self.test_X = np.matrix(test_batch['data'])
        self.test_y = np.matrix(test_batch['labels'])
        return 
    
    # reads in data from matlab file: fname, converts it into the wanted format and saves it to new_fname. 
    def separate_data(self, fname, new_fname):
        #data_batch = loadmat('../Datasets/cifar-10-batches-mat/' + fname)
        data_batch = loadmat(self.data_path + fname)
        x = np.float64(np.matrix(data_batch['data']).transpose())    # image pixel data (d x n)
        y = np.matrix(np.array(data_batch['labels'])) # labels for each of the n images. An index 0-9
        # create a one-hot representation of Y.
        y_m = np.zeros((self.K, x.shape[1]), dtype=np.float64)
        for i in range(x.shape[1]):
            j = y[i][0] # label image i
            y_m[j,i] = 1
        scipy.io.savemat('../Datasets/'+new_fname, dict(data=x, onehot = y_m, labels = y))
        return dict(data=x, onehot = y_m, labels = y)


    @staticmethod
    def normalize_data(X, mean_X, std_X):
        X = X - numpy.matlib.repmat(mean_X, 1, np.size(X,1) )
        X = np.divide(X, numpy.matlib.repmat(std_X, 1, np.size(X,1) ))
        return X

    def pre_process(self):
        # find mean and std of training data
        mean_trainX  = np.mean(self.train_X, axis=1)#np.matrix(self.train_X.mean(1)).transpose()
        std_trainX   = np.std(self.train_X, axis=1)#np.matrix(self.train_X.std(1)).transpose()

        # Normalize training, validation and test datasets
        self.train_X = self.normalize_data(self.train_X, mean_trainX, std_trainX)
        self.val_X = self.normalize_data(self.val_X, mean_trainX, std_trainX)
        self.test_X = self.normalize_data(self.test_X , mean_trainX, std_trainX)
        return 

    def set_net_dimensions(self):
        self.d = self.train_X.shape[0]
        self.n = self.train_X.shape[1]
        self.layer_dims = [self.d] + self.layer_dims + [self.K]
        if self.net_hyper_params['n_step'] is None:
            self.net_hyper_params['n_step'] = 2 * self.n//self.net_hyper_params['batch_size']
        
        return
    
    def init_parameters(self, init_type, sigma = None):
        self.net_params['W']      = [None] * self.k
        self.net_params['b']      = [None] * self.k
        self.net_params['betas']  = [None] * self.k
        self.net_params['gammas'] = [None] * self.k
        
        for l in range(1,self.k+1):
                # Xavier
                if init_type == "Xavier":
                    self.net_params['W'][l-1] = np.random.normal(loc=0.0, scale=1/math.sqrt(self.layer_dims[l-1]), size=(self.layer_dims[l],self.layer_dims[l-1]))
                
                elif init_type == "He":
                    self.net_params['W'][l-1] = np.random.normal(loc=0.0, scale=math.sqrt(2/self.layer_dims[l-1]), size=(self.layer_dims[l],self.layer_dims[l-1]))
                
                elif init_type == "Sigma":
                    self.net_params['W'][l-1] = np.random.normal(loc=0.0, scale = sigma, size=( self.layer_dims[l], self.layer_dims[l-1]))            
                
                self.net_params['b'] [l-1] = np.zeros((self.layer_dims[l],1), dtype=np.float64)

                if self.net_options['batch_norm']:
                    self.net_params['gammas'][l-1] = np.ones((self.layer_dims[l],1))
                    self.net_params['betas'][l-1] = np.zeros((self.layer_dims[l], 1))

    def compute_cost(self, X, Y, net_params, lamdba_, mode='training'):
        P, _, _ = self.evaluate_classifier(X, net_params, mode)
        sum_lcross = 0
        for col_idx in range(X.shape[1]):
            y, p = Y[:,col_idx] , P[:,col_idx]
            sum_lcross += -np.dot(y.T, np.log(p))[0,0]

        sum_square_weights = 0
        for l in range(self.k):
            sum_square_weights += np.sum(np.square(net_params['W'][l]), dtype=np.float64)
        
        regularization = lamdba_ * sum_square_weights
        loss = 1/X.shape[1] * sum_lcross 
        cost = loss + regularization

        return cost, loss
    
    @staticmethod
    def softmax(x):
        """ Standard definition of the softmax function """
        return np.exp(x) / np.sum(np.exp(x), axis=0)
    
    @staticmethod
    def batch_normalize(s, mean_s, var_s):
        var_s = np.array(var_s)
        return np.diag(np.power(var_s + np.finfo(float).eps, -0.5).T[0]) @ (s-mean_s)


    def evaluate_classifier(self, X_batch, net_params, mode='training'):
        n_b = X_batch.shape[1]
        if self.net_options['batch_norm']:
            if mode == 'training':
                mean_s = [None] * (self.k ) # real length is k-1
                var_s = [None] * (self.k ) # real length is k-1
            else: 
                mean_s = None
                var_s = None
            s = [None] * self.k
            s_circum = [None] * (self.k) # real length is k-1
            s_tilde  = [None] * (self.k) # real length is k-1
            H = [ X_batch ]                
            for l in range(1, self.k):
                s[l-1] = (net_params['W'][l-1] @ H[l-1]) + net_params['b'][l-1]                    
                if mode == 'training':
                    mean_s[l-1] = np.mean(s[l-1], axis=1, dtype=np.float64)
                    var_s[l-1]  = np.var(s[l-1], axis=1, dtype=np.float64)
                    if self.is_s_averages_init: 
                        self.mean_s_av[l-1] = self.net_hyper_params['alpha'] * self.mean_s_av[l-1] + (1-self.net_hyper_params['alpha'])*mean_s[l-1]
                        self.var_s_av[l-1]  = self.net_hyper_params['alpha'] * self.var_s_av[l-1] + (1-self.net_hyper_params['alpha'])*var_s[l-1]
                    s_circum[l-1] = self.batch_normalize(s[l-1], mean_s[l-1], var_s[l-1])
                if mode == 'testing':
                     s_circum[l-1] = self.batch_normalize(s[l-1], self.mean_s_av[l-1],self.var_s_av[l-1])
                s_tilde[l-1] = np.multiply(net_params['gammas'][l-1], s_circum[l-1]) + net_params['betas'][l-1]
                H.append( np.maximum(s_tilde[l-1], np.zeros(s_tilde[l-1].shape, dtype=np.float64)) )
            if not self.is_s_averages_init : 
                self.mean_s_av = mean_s
                self.var_s_av = var_s

            s[-1] = net_params['W'][-1] @ H[-1] + net_params['b'][-1]

            P = self.softmax( s[-1] )
            return P, H, {'mean_s':mean_s, 'var_s':var_s, 's':s, 's_circum': s_circum }    

        else:
            H = [ X_batch ]
            for l in range(1, self.k):
                s = (net_params['W'][l-1] @ H[l-1]) + np.dot(net_params['b'][l-1],  np.ones((1,n_b) , dtype=np.float64))
                H.append( np.maximum(s, np.zeros(s.shape, dtype=np.float64)) )
            P = self.softmax(net_params['W'][-1] @ H[-1] + np.dot(net_params['b'][-1], np.ones((1,n_b) , dtype=np.float64)) )
            return P, H, None   


    def batch_norm_back_pass(self, G_batch, s_batch, s_mean, s_var):
        n_b = G_batch.shape[1]
        sigma_1 = (np.power(s_var + np.finfo(float).eps,-0.5))
        sigma_2 = (np.power(s_var + np.finfo(float).eps,-1.5))

        G_1 = np.multiply(G_batch, np.dot(sigma_1, np.ones((1,n_b), dtype=np.float64)))
        G_2 = np.multiply(G_batch, np.dot(sigma_2, np.ones((1,n_b), dtype=np.float64)))

        D = s_batch - np.dot(s_mean, np.ones((1,n_b), dtype=np.float64))
        c = np.multiply(G_2, D) @ np.ones((n_b,1), dtype=np.float64)
        return G_1 - 1/n_b * (G_1 @ np.ones((n_b,1), dtype=np.float64)) @ np.ones((1,n_b), dtype=np.float64) - 1/n_b * np.multiply(D, c@np.ones((1,n_b), dtype=np.float64))


    def compute_gradients(self, X_batch, Y_batch, lambda_, data_subset = None, mode='training'):
        k = self.k

        if self.net_options['gradient_test']:
            X_batch, Y_batch = X_batch[:, :data_subset], Y_batch[:, :data_subset]

        n_b = X_batch.shape[1]
        
        #forward pass
        P_batch, H_batch, scores = self.evaluate_classifier(X_batch, self.net_params, mode)

        self.net_grads['W'] = [None] * k
        self.net_grads['b'] = [None] * k

        # Propagate the gradient through the loss and softmax operations
        G_batch = -(Y_batch - P_batch)

        if self.net_options['batch_norm']:
            
            self.net_grads['gammas'] = [None] * k
            self.net_grads['betas']  = [None] * k

            s, s_mean, s_var, s_circum = scores['s'], scores['mean_s'], scores['var_s'], scores['s_circum']

            # Gradients of J w.r.t. bias vector b[k-1] and W[k-1]:

            self.net_grads['W'][-1] = 1/n_b * (G_batch @ H_batch[-1].T) + 2 * lambda_ * self.net_params['W'][-1]
            self.net_grads['b'][-1] = 1/n_b * (G_batch @ np.ones((n_b,1) , dtype=np.float64)) 
            # TODO: FEL härnedan
            # Propagate G_batch to the previous layer
            G_batch =  self.net_params['W'][-1].T @ G_batch
            G_batch = np.multiply(G_batch, np.where(H_batch[-1] > 0, 1, 0))

            for l in range(k-1, 0, -1):
                # Gradients for the scale and offset parameters for layer l
                self.net_grads['gammas'][l-1] = 1/n_b * (np.multiply(G_batch, s_circum[l-1])) @ np.ones((n_b,1) , dtype=np.float64)
                self.net_grads['betas'][l-1]  = 1/n_b * G_batch @ np.ones((n_b,1) , dtype=np.float64)

                # Propagate the gradients through the scale and shift
                G_batch = np.multiply(G_batch, self.net_params['gammas'][l-1] @ np.ones((1,n_b), dtype=np.float64))

                # Propagate G_batch through the batch normalization
                G_batch = self.batch_norm_back_pass(G_batch, s[l-1], s_mean[l-1], s_var[l-1]) # TODO tror att det är fel här 

                # Gradients of J w.r.t. bias vector b[l-1] and W[l-1]:
                self.net_grads['W'][l-1] = 1/n_b * (G_batch @ H_batch[l-1].T) + 2 * lambda_ * self.net_params['W'][l-1]
                self.net_grads['b'][l-1] = 1/n_b * (G_batch @ np.ones((n_b,1) , dtype=np.float64)) 

                if l > 1: 
                    #propagate Gbatch to the previous layer
                    G_batch = self.net_params['W'][l-1].T @ G_batch
                    G_batch = np.multiply(G_batch, np.where(H_batch[l-1] > 0, 1, 0))


        else:
            for l in range(k, 1, -1): 
                self.net_grads['W'][l-1] = 1/n_b * (G_batch @ H_batch[l-1].T) + 2 * lambda_ * self.net_params['W'][l-1]
                self.net_grads['b'][l-1] = 1/n_b * (G_batch @ np.ones((n_b,1) , dtype=np.float64)) 

                G_batch = self.net_params['W'][l-1].T @ G_batch
                G_batch = np.multiply(G_batch, np.where(H_batch[l-1] > 0, 1, 0))

            self.net_grads['W'][0] = 1/n_b * (G_batch @ X_batch.T) + 2*lambda_ * self.net_params['W'][0] 
            self.net_grads['b'][0] = 1/n_b * (G_batch @ np.ones((n_b,1) , dtype=np.float64)) 
        
        return self.net_grads


    def compute_grads_num_slow(self, X, Y, lambda_, h, data_subset = None):
        """ Converted from matlab code """
        k = self.k
        num_grads = {'W'     : [None] * k, 'b'    : [None] * k,
                     'gammas': [None] * k, 'betas': [None] * k   }

        if self.net_options['gradient_test']:
            X, Y = X[:, :data_subset], Y[:, :data_subset]

        for l in range(k):
            num_grads['b'][l] = np.zeros(self.net_params['b'][l].shape , dtype=np.float64)
            net_params_try = copy.deepcopy(self.net_params)
            for i in range(len(self.net_params['b'][l])):
                b_try = copy.deepcopy(self.net_params['b'])
                b_try[l][i] -= h
                net_params_try['b'] = b_try
                c1, _ = self.compute_cost(X, Y, net_params_try, lambda_)

                b_try = copy.deepcopy(self.net_params['b'])
                b_try[l][i] += h
                net_params_try['b'] = b_try
                c2, _ = self.compute_cost(X, Y, net_params_try, lambda_)

                num_grads['b'][l][i] = (c2-c1) / (2*h)

            num_grads['W'][l] = np.zeros(self.net_params['W'][l].shape , dtype=np.float64)
            net_params_try = copy.deepcopy(self.net_params)
            for i in np.ndindex(self.net_params['W'][l].shape):
                W_try = copy.deepcopy(self.net_params['W'])
                W_try[l][i] -= h
                net_params_try['W'] = W_try
                c1, _ = self.compute_cost(X, Y, net_params_try, lambda_)

                W_try = copy.deepcopy(self.net_params['W'])
                W_try[l][i] += h
                net_params_try['W'] = W_try
                c2, _ = self.compute_cost(X, Y, net_params_try, lambda_)

                num_grads['W'][l][i] = (c2-c1) / (2*h)
        
            if self.net_options['batch_norm']:
                
                num_grads['gammas'][l] = np.zeros(self.net_params['gammas'][l].shape, dtype=np.float64)
                net_params_try = copy.deepcopy(self.net_params)
                for i in np.ndindex(self.net_params['gammas'][l].shape): 
                    gammas_try = copy.deepcopy(self.net_params['gammas'])
                    gammas_try[l][i] -= h
                    net_params_try['gammas'] = gammas_try
                    c1, _ = self.compute_cost(X, Y, net_params_try, lambda_)

                    gammas_try = copy.deepcopy(self.net_params['gammas'])
                    gammas_try[l][i] += h
                    net_params_try['gammas'] = gammas_try
                    c2, _ = self.compute_cost(X, Y, net_params_try, lambda_)
                    
                    num_grads['gammas'][l][i] = (c2-c1) / (2*h)
                
                num_grads['betas'][l] = np.zeros(self.net_params['betas'][l].shape, dtype=np.float64)
                net_params_try = copy.deepcopy(self.net_params)
                for i in np.ndindex(self.net_params['betas'][l].shape): 
                    betas_try = copy.deepcopy(self.net_params['betas'])
                    betas_try[l][i] -= h
                    net_params_try['betas'] = betas_try
                    c1, _ = self.compute_cost(X, Y, net_params_try, lambda_)

                    betas_try = copy.deepcopy(self.net_params['betas'])
                    betas_try[l][i] += h
                    net_params_try['betas'] = betas_try
                    c2, _ = self.compute_cost(X, Y, net_params_try, lambda_)
                    
                    num_grads['betas'][l][i] = (c2-c1) / (2*h)

        return num_grads


    def mini_batch_GD(self, train_X=None, train_Y=None, train_y=None, val_X=None, val_Y=None, val_y=None):

        # if not specific data is given, use the networks data
        if all(var is None for var in [train_X, train_Y, train_y, val_X, val_Y,val_y]):
            train_X, train_Y, train_y, val_X, val_Y, val_y = self.train_X, self.train_Y, self.train_y, self.val_X, self.val_Y, self.val_y
        
        lambda_   = self.net_hyper_params['lambda_']
        n_batch = self.net_hyper_params['batch_size']
        cycles  = self.net_hyper_params['cycles']

        eta_min = self.net_hyper_params['eta_min']
        eta_max = self.net_hyper_params['eta_max']
        n_step =  self.net_hyper_params['n_step']

        n_epoch = int((cycles * 2 * n_step ) / (train_X.shape[1] / n_batch)) # num of epochs to fit the requested amount of cycles
        
        plt_data = {
                    'etas'              : [],
                    'val_accuracies'    : [],
                    'train_accuracies'  : [],
                    'training_costs'    : [],
                    'training_losses'   : [],
                    'validation_costs'  : [],
                    'validation_losses' : []   }
        
        t = 0
        eta_t = eta_min
        for _ in range(n_epoch):
            if self.net_options['rand_batch']:
                r_idx = np.random.permutation(train_X.shape[1])
            for j in range(train_X.shape[1] // n_batch):
                start = j * n_batch
                end = (j+1) * n_batch
                if self.net_options['rand_batch']:
                    rand_range = r_idx[range(start,end)]
                    X_batch = train_X[:, rand_range] 
                    Y_batch = train_Y[:, rand_range] 
                else:
                    X_batch = train_X[:, start:end] 
                    Y_batch = train_Y[:, start:end] 

                self.compute_gradients(X_batch, Y_batch, lambda_)
                self.is_s_averages_init = True
                for l in range(self.k):
                    self.net_params['W'][l] -= eta_t * self.net_grads['W'][l]
                    self.net_params['b'][l] -= eta_t * self.net_grads['b'][l]
                    if self.net_options['batch_norm'] and l < self.k-1:
                        self.net_params['gammas'][l] -= eta_t * self.net_grads['gammas'][l]
                        self.net_params['betas'][l] -= eta_t * self.net_grads['betas'][l]
                    
                t+=1
                cycle = math.floor(t/(2*n_step)) # num of cycles done
                plt_data['etas'].append(eta_t)
        
                if (t <= (2 * cycle + 1) * n_step) and (t >=2 * cycle * n_step):
                    eta_t = eta_min+((t - (2 * cycle * n_step)) / n_step)*(eta_max - eta_min)
                elif (t <= 2 * (cycle + 1) * n_step) and (t >= (2 * cycle + 1) * n_step) :
                    eta_t = eta_max-((t - ((2 * cycle + 1) * n_step)) / n_step)*(eta_max - eta_min)
                
                if not self.net_options['lmd_search'] and self.net_options['plotting']:
                    # Compute loss, cost, accuracy for each iteration and store in arrays
                    if( t % ((2*n_step)/10) == 0 ):
                        train_acc = self.compute_accuracy(train_X, train_y, mode = 'testing')
                        val_acc   = self.compute_accuracy(val_X, val_y, mode = 'testing')
                        plt_data['train_accuracies'].append( train_acc )
                        plt_data['val_accuracies'].append( val_acc )
                        train_cost, train_loss = self.compute_cost(train_X, train_Y, self.net_params, self.net_hyper_params['lambda_'], mode = 'testing')
                        plt_data['training_costs'].append( train_cost )
                        plt_data['training_losses'].append( train_loss )

                        val_cost, val_loss = self.compute_cost(val_X, val_Y, self.net_params, self.net_hyper_params['lambda_'], mode = 'testing')
                        plt_data['validation_costs'].append( val_cost )
                        plt_data['validation_losses'].append( val_loss )
                
        return plt_data
    
    def perform_training(self, log_time = True):
        t_start = time.time()
        plot_results = self.mini_batch_GD()
        if log_time:
            print('Training w/ Mini-Batch GD took ' + str(round(time.time()-t_start,2)) +  's')
        return plot_results

    def compute_accuracy(self, X, y, mode='training'):
        # compute models predictions in vector:
        P, _, _ = self.evaluate_classifier(X, self.net_params, mode)
        amount_of_correct = 0
        #for each image in dataset X:
        for img in range(P.shape[1]):
            # find index of the highest prob of the p vector for each image.
            prediction = np.where(P[:,img] == max(P[:,img]))[0][0]
            if prediction == y[img][0]:
                amount_of_correct += 1
        return amount_of_correct / y.shape[0]

    def lamdba_search(self, iter=20):
 
        ## COARSE SEARCH
        coarse_lamdas = np.arange(-5, -1, 0.5)
        val_accuracies = []
        for lamb in coarse_lamdas:
            self.init_parameters(init_type="He")
            self.set_hyper_params(lambda_= 10**lamb)
            self.mini_batch_GD()
            val_accuracies.append( self.compute_accuracy(self.val_X, self.val_y) )
            
        #Plot results
        plt.scatter([10**q for q in coarse_lamdas], val_accuracies)
        plt.xlabel('lambda')
        plt.ylabel('val accuracy')
        plt.savefig('Result_Pics/coarse_seach.png')
        plt.close()
        print(val_accuracies)
        print([10**q for q in coarse_lamdas])

        ### FINE RANDOM SEARCH
        best_3_lamb = np.take(coarse_lamdas, np.argsort(val_accuracies)[-3:])

        l_min, l_max = min(best_3_lamb), max(best_3_lamb)
        accs = []
        lambs = []
        for _ in range(iter):
            self.init_parameters(init_type="He")
            l = l_min + (l_max - l_min) * random.uniform(0,1)
            lambs.append(l)
            self.set_hyper_params(lambda_= 10**l) 
            self.mini_batch_GD()
            accs.append( self.compute_accuracy(self.val_X, self.val_y) )

        print(accs)
        print(lambs)
        best_3_lamb = np.take(lambs, np.argsort(accs)[-3:])
        print("The 3 best lambda values:", best_3_lamb, "and their respective accuracies on validation dataset",np.take(accs, np.argsort(accs)[-3:]))

        #Plot results
        plt.scatter([10**q for q in lambs], accs)
        plt.xlabel('lambda')
        plt.ylabel('val accuracy')
        plt.savefig('Result_Pics/fine_seach.png')
        plt.close()
        self.lambda_srch_res = (10 ** np.take(lambs, np.argsort(accs)[-1:])[0] , np.take(accs, np.argsort(accs)[-1:])[0])
        return self.lambda_srch_res

        

class NN_Plotting():
    def __init__(self, neural_network, result_path = 'Result_Pics/'):
        self.nn = neural_network
        self.result_path = result_path
    
    def plot_cost_loss_acc(self, train_costs, val_costs, train_losses, val_losses,train_accuracies, val_accuracies, fig_idx):

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12,3))

        ax1.plot(range(len(val_costs)), val_costs, label= 'Validation', color= 'Red')
        ax1.plot(range(len(train_costs)), train_costs, label= 'Training', color= 'Green')
        ax1.set_title('Cost plot')
        ax1.set(xlabel='Time', ylabel='Cost')

        ax2.plot(range(len(val_losses)), val_losses, label= 'Validation ', color= 'Red')
        ax2.plot(range(len(train_losses)), train_losses, label= 'Training', color= 'Green')
        ax2.set_title('Loss plot')
        ax2.set(xlabel='Time', ylabel='Loss')

        ax3.plot(range(len(val_accuracies)), val_accuracies, label= 'Validation', color= 'Red')
        ax3.plot(range(len(train_accuracies)), train_accuracies, label= 'Training', color= 'Green')
        ax3.set_title('Accuracy plot')
        ax3.set(xlabel='Time', ylabel='Accuracy')

        fig.tight_layout()
        plt.legend()
        plt.close()
        #plt.show()
        fig_name = self.result_path + 'Fig_' + fig_idx + '.png'
        fig.savefig(fig_name, bbox_inches='tight')
        
        #add used parameters as meta text of png file
        img = Image.open(fig_name)
        metadata = PngInfo()
        metadata.add_text('network options', str(self.nn.net_options))
        metadata.add_text('network hyperparams',  str(self.nn.net_hyper_params))
        metadata.add_text('layer_dims', str(self.nn.layer_dims)) 
        metadata.add_text('number of layers', str(self.nn.k) )
        img.save(fig_name, pnginfo=metadata)


    def plot_learning_rates(self, etas):

        plt.plot(range(len(etas)), etas, label= 'Cyclic learning rate' , color= 'Blue')
        plt.plot(range(len(etas)), [self.eta_max for i in range(len(etas))], label= 'eta_max' , color= 'Red')
        plt.plot(range(len(etas)), [self.eta_min for i in range(len(etas))], label= 'eta_min' , color= 'Green')
        plt.xlabel("Update steps")
        plt.ylabel('eta_t')
        plt.legend()
        #plt.show()
        plt.savefig(self.result_path + 'cyclic_lr_w_' + str(self.net_params['cycles']) + '_cycles' + + '.png', bbox_inches='tight')
        plt.close() 

    def plot_loss(self, train_losses, val_losses, fig_name):
        plt.plot(range(len(val_losses)), val_losses, label= 'Validation ', color= 'Red')
        plt.plot(range(len(train_losses)), train_losses, label= 'Training', color= 'Green')
        plt.xlabel('Time')
        plt.ylabel('Loss')

        plt.legend()
        #plt.show()
        fig_name = self.result_path + fig_name + '.png'
        plt.savefig(fig_name, bbox_inches='tight')
        plt.close()

        #add used parameters as meta text of png file
        img = Image.open(fig_name)
        metadata = PngInfo()
        metadata.add_text('network options', str(self.nn.net_options))
        metadata.add_text('network hyperparams',  str(self.nn.net_hyper_params))
        metadata.add_text('layer_dims', str(self.nn.layer_dims)) 
        metadata.add_text('number of layers', str(self.nn.k) )
        img.save(fig_name, pnginfo=metadata)

        return 0

def main():
    EXE1     = False
    EXE2_pt1 = False
    EXE2_pt2 = True
    EXE3_pt1 = True
    EXE3_pt2 = True
    EXE3_pt3 = True
    EXE3_pt4 = True
    EXE3_pt5 = True

    """Exercise 1 TESTING CORRECTNESS OF GRADIENTS """
    if EXE1:
        nn = NeuralNetwork(k = 4, K=10, layer_dims = [50, 50, 50], param_init_type = 'Xavier', val_data_size=None)
        nn.set_options(gradient_test = True)
        nn.set_hyper_params()
        print("\n=== Exercise 1: Testing Gradients ===")
        W_grads, b_grads = nn.compute_gradients(nn.train_X, nn.train_Y, nn.net_hyper_params['lambda_'], data_subset=10)
        W_grads_num, b_grads_num = nn.compute_grads_num_slow(nn.train_X, nn.train_Y, nn.net_hyper_params['lambda_'], 1e-5, data_subset=10)
        grad_diffs = PrettyTable(['Layer', 'diff grad_W', 'diff grad_b'])
        for layer in range(nn.k):
            # Test that numerical and analytical gradient computations are the same
            diff_W = np.linalg.norm((W_grads[layer]) - np.matrix(W_grads_num[layer])) / max( 1e-6, np.linalg.norm((W_grads[layer])) + np.linalg.norm((W_grads_num[layer])))
            diff_b = np.linalg.norm((b_grads[layer]) - np.matrix(b_grads_num[layer])) / max( 1e-6, np.linalg.norm((b_grads[layer])) + np.linalg.norm((b_grads_num[layer])))
            grad_diffs.add_row([layer+1, diff_W, diff_b])
            np.testing.assert_almost_equal(np.matrix(W_grads[layer]), np.matrix(W_grads_num[layer]), decimal=6)
            np.testing.assert_almost_equal(np.matrix(b_grads[layer]), np.matrix(b_grads_num[layer]), decimal=6)
        print(grad_diffs)

    """Exercise 2"""
    if EXE2_pt1:
        ### FIRST CHECK: replicate default result of Assignment 2 with a 2-layered network
        # Replica of Fig 3
        print("\n=== Exercise 2 ===")
        print("--- Replicate results of Figure 3 from Assignment 2 ---")
        nn = NeuralNetwork(k = 2, K=10, layer_dims = [50], param_init_type = 'Xavier', val_data_size=None)
        nn.set_hyper_params(cycles=1, batch_size=100, lambda_=0.01, n_step = 500)
        nn.set_options(plotting=True)
        plt_data = nn.perform_training()
        # test the accuracy of model on test dataset
        acc = nn.compute_accuracy(nn.test_X, nn.test_y) * 100
        print("Accuracy of testdata = " + str(acc) + "%")

        nn_plt = NN_Plotting(nn)
        nn_plt.plot_cost_loss_acc(
                                plt_data['training_costs'],   plt_data['validation_costs'],
                                plt_data['training_losses'],plt_data['validation_losses'],
                                plt_data['train_accuracies'], plt_data['val_accuracies'], 'rep_3_with_acc_' + str(round(acc,2)).replace('.', '')
                                )
        # Replica of Fig 4
        print("--- Replicate results of Figure 4 from Assignment 2 ---")
        nn = NeuralNetwork(k = 2, K=10, layer_dims = [50], param_init_type = 'Xavier', val_data_size=None)
        nn.set_hyper_params(cycles=3, batch_size=100, lambda_=0.01, n_step = 800)
        nn.set_options(plotting=True)
        plt_data = nn.perform_training()
        # test the accuracy of model on test dataset
        acc = nn.compute_accuracy(nn.test_X, nn.test_y) * 100
        print("Accuracy of testdata = " + str(acc) + "%")

        nn_plt = NN_Plotting(nn)
        nn_plt.plot_cost_loss_acc(
                                plt_data['training_costs'],   plt_data['validation_costs'],
                                plt_data['training_losses'],plt_data['validation_losses'],
                                plt_data['train_accuracies'], plt_data['val_accuracies'], 'rep_4_with_acc_' + str(round(acc,2)).replace('.', '')
                                )

    if EXE2_pt2:
        ### SECOND CHECK: 
        
        print('--- 3 layered network w/o BN ---')
        # 3-layered network with 50 nodes in hidden layers
        nn = NeuralNetwork(k = 3, K=10, layer_dims = [50,50], param_init_type = 'He', val_data_size=5000, plotting=False, rand_batch = True)
        nn.set_hyper_params(cycles=2, batch_size=100, lambda_=0.005, n_step = 5*450)
        plt_data = nn.perform_training()
        if nn.net_options['plotting']:
            nn_plt = NN_Plotting(nn)
            nn_plt.plot_loss(train_losses=plt_data['training_losses'], val_losses=plt_data['validation_losses'], fig_name='3lay_no_BN')

        # test the accuracy of model on test dataset
        acc = nn.compute_accuracy(nn.test_X, nn.test_y, mode='testing') * 100
        print("Accuracy of testdata = " + str(acc) + "%")
        
        # 9-layered network with nodes per layer [50, 30, 20, 20, 10, 10, 10, 10]
        print('--- 9 layered network w/o BN ---')
        nn = NeuralNetwork(k = 9, K=10, layer_dims = [50, 30, 20, 20, 10, 10, 10, 10], param_init_type = 'He', val_data_size=5000, plotting=False, rand_batch = True )
        nn.set_hyper_params(cycles=2, batch_size=100, lambda_=0.005, n_step = 5*450)
        plt_data = nn.perform_training()
        if nn.net_options['plotting']:
            nn_plt = NN_Plotting(nn)
            nn_plt.plot_loss(train_losses=plt_data['training_losses'], val_losses=plt_data['validation_losses'], fig_name='9lay_no_BN')

        # test the accuracy of model on test dataset
        acc = nn.compute_accuracy(nn.test_X, nn.test_y, mode='testing') * 100
        print("Accuracy of testdata = " + str(acc) + "%")


    """Exercise 3: Batch Normalization implementation"""
    if EXE3_pt1:
        """Check gradient computations with BN"""
        print('\n=== Exercise 3: Batch Normalization ===')
        print('--- Checking gradient computations for 2-layered nn w/ BN ---')
        nn = NeuralNetwork(k = 2, K=10, layer_dims = [50], param_init_type = 'He', batch_norm=True)
        nn.set_options(gradient_test = True) 
        nn.set_hyper_params()
        grads_analytic = nn.compute_gradients(nn.train_X, nn.train_Y, nn.net_hyper_params['lambda_'], data_subset=100)
        grads_numeric  = nn.compute_grads_num_slow(nn.train_X, nn.train_Y, nn.net_hyper_params['lambda_'], 1e-5, data_subset=100)
        grad_diffs = PrettyTable(['Layer', 'diff grad_W', 'diff grad_b', 'diff grad_gammas', 'diff grad_betas'])
        for layer in range(nn.k):
            # Calculate relative difference between  numerical and analytical gradient computations
            diff_W = np.linalg.norm((grads_analytic['W'][layer]) - np.matrix(grads_numeric['W'][layer])) / max( 1e-6, np.linalg.norm((grads_analytic['W'][layer])) + np.linalg.norm((grads_numeric['W'][layer])))
            diff_b = np.linalg.norm((grads_analytic['b'][layer]) - np.matrix(grads_numeric['b'][layer])) / max( 1e-6, np.linalg.norm((grads_analytic['b'][layer])) + np.linalg.norm((grads_numeric['b'][layer])))
            if layer != nn.k-1:
                diff_gammas = np.linalg.norm((grads_analytic['gammas'][layer]) - np.matrix(grads_numeric['gammas'][layer])) / max( 1e-6, np.linalg.norm((grads_analytic['gammas'][layer])) + np.linalg.norm((grads_numeric['gammas'][layer])))
                diff_betas = np.linalg.norm((grads_analytic['betas'][layer]) - np.matrix(grads_numeric['betas'][layer])) / max( 1e-6, np.linalg.norm((grads_analytic['betas'][layer])) + np.linalg.norm((grads_numeric['betas'][layer])))
                np.testing.assert_almost_equal(np.matrix(grads_analytic['gammas'][layer]), np.matrix(grads_numeric['gammas'][layer]), decimal=6)
                np.testing.assert_almost_equal(np.matrix(grads_analytic['betas'][layer]), np.matrix(grads_numeric['betas'][layer]), decimal=6)
            else: 
                diff_gammas, diff_betas = None, None

            grad_diffs.add_row([layer+1, diff_W, diff_b, diff_gammas, diff_betas])
            np.testing.assert_almost_equal(np.matrix(grads_analytic['W'][layer]), np.matrix(grads_numeric['W'][layer]), decimal=6)
            np.testing.assert_almost_equal(np.matrix(grads_analytic['b'][layer]), np.matrix(grads_numeric['b'][layer]), decimal=6)
            
        print(grad_diffs)

        print('--- Checking gradient computations for 3-layered nn w/ BN ---')
        nn = NeuralNetwork(k = 3, K=10, layer_dims = [50,50], param_init_type = 'He', batch_norm=True)
        nn.set_options(gradient_test = True) 
        nn.set_hyper_params()
        grads_analytic = nn.compute_gradients(nn.train_X, nn.train_Y, nn.net_hyper_params['lambda_'], data_subset=100)
        grads_numeric  = nn.compute_grads_num_slow(nn.train_X, nn.train_Y, nn.net_hyper_params['lambda_'], 1e-5, data_subset=100)
        grad_diffs = PrettyTable(['Layer', 'diff grad_W', 'diff grad_b', 'diff grad_gammas', 'diff grad_betas'])
        for layer in range(nn.k):
            # Calculate relative difference between  numerical and analytical gradient computations
            diff_W = np.linalg.norm((grads_analytic['W'][layer]) - np.matrix(grads_numeric['W'][layer])) / max( 1e-6, np.linalg.norm((grads_analytic['W'][layer])) + np.linalg.norm((grads_numeric['W'][layer])))
            diff_b = np.linalg.norm((grads_analytic['b'][layer]) - np.matrix(grads_numeric['b'][layer])) / max( 1e-6, np.linalg.norm((grads_analytic['b'][layer])) + np.linalg.norm((grads_numeric['b'][layer])))
            if layer != nn.k-1:
                diff_gammas = np.linalg.norm((grads_analytic['gammas'][layer]) - np.matrix(grads_numeric['gammas'][layer])) / max( 1e-6, np.linalg.norm((grads_analytic['gammas'][layer])) + np.linalg.norm((grads_numeric['gammas'][layer])))
                diff_betas = np.linalg.norm((grads_analytic['betas'][layer]) - np.matrix(grads_numeric['betas'][layer])) / max( 1e-6, np.linalg.norm((grads_analytic['betas'][layer])) + np.linalg.norm((grads_numeric['betas'][layer])))
                np.testing.assert_almost_equal(np.matrix(grads_analytic['gammas'][layer]), np.matrix(grads_numeric['gammas'][layer]), decimal=6)
                np.testing.assert_almost_equal(np.matrix(grads_analytic['betas'][layer]), np.matrix(grads_numeric['betas'][layer]), decimal=6)
            else: 
                diff_gammas, diff_betas = None, None
            grad_diffs.add_row([layer+1, diff_W, diff_b, diff_gammas, diff_betas])
            np.testing.assert_almost_equal(np.matrix(grads_analytic['W'][layer]), np.matrix(grads_numeric['W'][layer]), decimal=6)
            np.testing.assert_almost_equal(np.matrix(grads_analytic['b'][layer]), np.matrix(grads_numeric['b'][layer]), decimal=6)
            
        print(grad_diffs)
        
    if EXE3_pt2:   
        print('--- 3 layered network w/ BN ---')
        nn = NeuralNetwork( k = 3, K=10, layer_dims = [50,50], param_init_type = 'He', val_data_size=5000, 
                            batch_norm=True, rand_batch = True, plotting=False)
        nn.set_hyper_params(cycles=2, batch_size=100, lambda_=0.005, n_step = 5*450)
        plt_data = nn.perform_training()
        
        if nn.net_options['plotting']:
            nn_plt = NN_Plotting(nn)
            nn_plt.plot_loss(train_losses=plt_data['training_losses'], val_losses=plt_data['validation_losses'], fig_name='3lay_BN')

        # test the accuracy of model on test dataset
        acc = nn.compute_accuracy(nn.test_X, nn.test_y, mode='testing') * 100
        print("Accuracy of testdata = " + str(acc) + "%")

        print('--- 9 layered network w/ BN ---')
        nn = NeuralNetwork( k = 9, K=10, layer_dims = [50, 30, 20, 20, 10, 10, 10, 10], param_init_type = 'He', val_data_size=5000, 
                            batch_norm=True, rand_batch = True, plotting=True)
        nn.set_hyper_params(cycles=2, batch_size=100, lambda_=0.005, n_step = 5*450)
        plt_data = nn.perform_training()
        if nn.net_options['plotting']:
            nn_plt = NN_Plotting(nn)
            nn_plt.plot_loss(train_losses=plt_data['training_losses'], val_losses=plt_data['validation_losses'], fig_name='9lay_BN')

        # test the accuracy of model on test dataset
        acc = nn.compute_accuracy(nn.test_X, nn.test_y, mode='testing') * 100
        print("Accuracy of testdata = " + str(acc) + "%")
        
    if EXE3_pt3:
        print('--- Coarse-to-fine lambda search ---')
        nn = NeuralNetwork( k = 3, K=10, layer_dims = [50,50], param_init_type = 'He', val_data_size=5000, 
                            batch_norm=True, lmd_search=True, rand_batch=True)
        nn.set_hyper_params(cycles=2, batch_size=100, n_step = 5*450)
        t_start = time.time()
        best_found = nn.lamdba_search(iter=20)
        print('lmd search took ', str(datetime.timedelta(seconds=time.time()-t_start)) , 's. Best lambda found: ', best_found[0], 'with accuracy ', best_found[1]*100)

    if EXE3_pt4:
        print('--- 3 layered network w/ BN & best found lambda---')
        nn = NeuralNetwork( k = 3, K=10, layer_dims = [50,50], param_init_type = 'He', val_data_size=5000, 
                            batch_norm=True, rand_batch = True, plotting=False)
        nn.set_hyper_params(cycles=2, batch_size=100, lambda_= 10**(-2.29795015), n_step = 5*450)
        nn.perform_training()
        
        # test the accuracy of model on test dataset
        acc = nn.compute_accuracy(nn.test_X, nn.test_y,mode='testing' ) * 100
        print("Accuracy of testdata = " + str(acc) + "%")

    
    """Exercise Sensitivity to initialization"""
    if EXE3_pt5:
        print('\n--- Sensitivity to initialization experiment ---')
        sigmas = [1e-1, 1e-3, 1e-4]
        for sig in sigmas:
            """ With BN """
            print('--- With BN --- ')
            nn = NeuralNetwork( k = 3, K=10, layer_dims = [50,50], param_init_type = 'Sigma', val_data_size=5000, 
                                batch_norm=True, rand_batch = True, sigma=sig, plotting=True)
            nn.set_hyper_params(cycles=2, batch_size=100, lambda_= 0.005, n_step = 2*450)
            plt_data = nn.perform_training()
            if nn.net_options['plotting']:
                nn_plt = NN_Plotting(nn)
                nn_plt.plot_loss(train_losses=plt_data['training_losses'], val_losses=plt_data['validation_losses'], fig_name='sensitivity_exp_BN_'+str(sig))

            # test the accuracy of model on test dataset
            acc = nn.compute_accuracy(nn.test_X, nn.test_y,mode='testing' ) * 100
            print("Accuracy of testdata = " + str(acc) + "%")

            """ Without BN """
            print('--- Without BN --- ')

            nn = NeuralNetwork( k = 3, K=10, layer_dims = [50,50], param_init_type = 'Sigma', val_data_size=5000, 
                                batch_norm=False, rand_batch = True, sigma=sig, plotting=True)
            nn.set_hyper_params(cycles=2, batch_size=100, lambda_= 0.005, n_step = 2*450 )
            plt_data = nn.perform_training()

            if nn.net_options['plotting']:
                nn_plt = NN_Plotting(nn)
                nn_plt.plot_loss(train_losses=plt_data['training_losses'], val_losses=plt_data['validation_losses'], fig_name='sensitivity_exp_no_BN_'+str(sig))

            # test the accuracy of model on test dataset
            acc = nn.compute_accuracy(nn.test_X, nn.test_y,mode='testing' ) * 100
            print("Accuracy of testdata = " + str(acc) + "%")



if __name__ == "__main__":
    main()