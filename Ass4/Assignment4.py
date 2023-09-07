__author__ = 'Léo Noharet'
__email__ = 'lnoharet@kth.se'
__date__ = '10/5/2022'

from cmath import inf
import copy
from prettytable import PrettyTable
from scipy.io import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
np.random.seed(0)

class RNN():
    def __init__(self, m = 100, lr=0.1, seq_length=25,  data_path = 'goblet_book.txt',
                 param_init_sig = 0.01, n_epoch = 10):

        self.data_path = data_path
        self.book_data = None
        self.book_chars = None
        self.char_to_ind = {}
        self.ind_to_char = {}

        self.X = None #  one-hot representation of data
        self.Y = None # one-hot representation of labels

        self.K = None # dimensionality of output
        self.m = m # num of nodes on hidden layer
        self.d = None # dimensionality of input
        self.lr = lr # learning rate
        self.n_epoch = n_epoch
        self.seq_length = seq_length # length of input sequence

        self.param_init_sig = param_init_sig
        self.net_params = None 
        self.best_net_params = None
        
        # read in data and create mappings char_to_ind and ind_to_char
        self.pre_process_data()

        # init network parameters
        self.init_parameters()

        # read input data and create mappings char_to_ind and ind_to_char
        self.pre_process_data()


    def pre_process_data(self):
        with open(self.data_path) as f:
                self.book_data = f.read()
        self.book_chars  = list(set(self.book_data)) # gets all unique chars of text
        self.K = len(self.book_chars)
        self.d = self.K
        # Create mapping char_to_ind and ind_to_char
        for c in self.book_chars:
            first_occ_idx = self.book_chars.index(c)
            self.char_to_ind[c] = first_occ_idx
            self.ind_to_char[first_occ_idx] = c


    def init_parameters(self):
        self.net_params = {
                            'b' : np.zeros((self.m, 1)),
                            'c' : np.zeros((self.K, 1)),
                            'U' : np.random.normal(loc=0, scale=self.param_init_sig, size =(self.m,self.d)), # weight matrix (m x d) input to hidden connection
                            'W' : np.random.normal(loc=0, scale=self.param_init_sig, size =(self.m,self.m)), # weight matrix (m x m) hidden to hidden connection
                            'V' : np.random.normal(loc=0, scale=self.param_init_sig, size =(self.K,self.m)) # weight matrix (K x m) hidden to output connection
                           }
        

    @staticmethod
    def softmax(x):
        """ Standard definition of the softmax function """
        return np.exp(x) / np.sum(np.exp(x), axis=0)


    def evaluate_classifier(self, h0, x, net_params = None):
        """
        Forward pass of the RNN. 
        :param h0: hidden state at time t = 0  
        :param x0: input vector at time t = 0
        :param seq_len: length of char sequence
        :return a_t, h_t, o_t, p_t
        """
        if net_params is None: net_params = self.net_params
        a_t = net_params['W'] @ h0 + net_params['U'] @ x  + net_params['b']
        h_t = np.tanh(a_t)
        o_t = net_params['V'] @ h_t + net_params['c']
        p_t = self.softmax(o_t)

        return a_t, h_t, o_t, p_t


    def synthesize_text(self, h0, x0, seq_len):
        """
        synthesizes a sequence of characters using the current parameter values of the RNN

        :param h0: hidden state at time 0  
        :param x0: dummy input vector (d x 1)
        :param n : length of seqence to generate
        :return Y: one-hot representation of text
        """
        Y = np.zeros((self.K,seq_len))
        sampled_chars = ''
        h_t = h0
        x_t = x0
        for t in range(seq_len):
            a_t = self.net_params['W'] @ h_t + self.net_params['U'] @ x_t + self.net_params['b']
            h_t = np.tanh(a_t)
            o_t = self.net_params['V'] @ h_t + self.net_params['c']
            p_t = self.softmax(o_t)
            # compute next input x_t by random sampling a character from p_t
            sampled_idx = np.random.choice(self.d, p = np.asarray(p_t).flatten()) #sample an id 0-d, with probability p
            x_t = np.zeros((self.d,1)) # next input character
            x_t[sampled_idx] = 1.0
            Y[:,t] = x_t.reshape((self.K,))
            sampled_chars += self.ind_to_char[sampled_idx]

        return Y, sampled_chars
    

    def convert_to_onehot(self, x):
        """ 
        :param x: input vector of length sequence length, a sequence of chars
        :return : one-hot matrix of input vector x
        """ 
        seq_length = len(x)
        Y = np.zeros((self.K, seq_length))
        for i in range(seq_length):
            Y[self.char_to_ind[x[i]],i] = 1
        return np.matrix(Y)


    def compute_cross_entropy_loss(self, X, Y, net_params):
        p = [None] * self.seq_length
        h = [None] * self.seq_length

        for t in range(self.seq_length):
            if t == 0:
                _, h[t], _, p[t] = self.evaluate_classifier(h0=np.zeros((self.m,1)), x=X[:,t], net_params=net_params)
            else:
                _, h[t], _, p[t] = self.evaluate_classifier(h0=h[t-1], x=X[:,t], net_params=net_params)
        
        sum_lcross = 0
        for t in range(self.seq_length):
            sum_lcross += -np.dot(Y[:,t].T, np.log(p[t]))[0,0]
        return sum_lcross 


    def compute_gradients(self, X,Y, h0 = None):

        if h0 is None: h0 = np.zeros((self.m, 1)) # hidden state at time t=0
        
        a_grad = [None] * self.seq_length
        h_grad = [None] * self.seq_length 
        o_grad = [None] * self.seq_length 
        a      = [None] * self.seq_length
        h      = [None] * self.seq_length
        o      = [None] * self.seq_length
        p      = [None] * self.seq_length
         
        gradients = {}
        for key, val in self.net_params.items():
            gradients[key] = np.zeros(val.shape)

        h[0] = h0
        loss = 0
        # forward pass
        for t in range(self.seq_length):
            if t == 0:
                a[t], h[t], o[t], p[t] = self.evaluate_classifier(h0=h0, x=X[:,t])
            else: 
                a[t], h[t], o[t], p[t] = self.evaluate_classifier(h0=h[t-1], x=X[:,t])
            loss += -np.dot(Y[:,t].T, np.log(p[t]))[0,0]

        # backward pass
        for t in range(self.seq_length):
            o_grad[t] = -(Y[:,t] - p[t]).T
            gradients['V'] += np.dot( o_grad[t].T, h[t].T )
            gradients['c'] += o_grad[t].T
        
        h_grad[-1] = o_grad[-1] @ self.net_params['V']
        a_grad[-1] =np.multiply(h_grad[-1], 1-np.power(np.tanh(a[-1].T), 2) )

        for t in range(self.seq_length-2, -1, -1):
            h_grad[t] = o_grad[t] @ self.net_params['V'] + a_grad[t+1] @ self.net_params['W']
            a_grad[t] = np.multiply(h_grad[t], 1-np.power(np.tanh(a[t].T),2))
        for t in range(self.seq_length):
            if t == 0:
                gradients['W'] += np.dot( a_grad[t].T, h0.T)
            else:
                gradients['W'] += np.dot( a_grad[t].T, h[t-1].T)
            
            gradients['U'] += np.dot( a_grad[t].T, X[:,t].T )
            gradients['b'] += a_grad[t].T
        
        for key, val in self.net_params.items():
            gradients[key] = np.clip(gradients[key], -5 ,5)
        
        return gradients, h[self.seq_length-1], loss


    def compute_grads_num(self, X, Y, h):

        num_grads = {}
        for key, val in self.net_params.items(): 
            num_grads[key] = np.zeros(val.shape)

        for key, val in num_grads.items(): 
            print('Computing numerical gradient for')
            print('Field name:', key )
            for i in np.ndindex(self.net_params[key].shape):

                params_try = copy.deepcopy(self.net_params)
                params_try[key][i] -= h
                l1 = self.compute_cross_entropy_loss(X, Y, params_try)

                params_try = copy.deepcopy(self.net_params)
                params_try[key][i] += h
                l2 = self.compute_cross_entropy_loss(X, Y, params_try)
                num_grads[key][i] = (l2-l1)/(2*h)
        
        for key, val in num_grads.items():
            num_grads[key] = np.clip(num_grads[key], -5 ,5)

        return num_grads
    

    def adagrad(self):

        smooth_loss = 0
        t_step = 0
        min_loss = (0,inf)
        smooth_losses = []
        sum_of_squared_grads = dict()
        
        for key, val in self.net_params.items():
            sum_of_squared_grads[key] = np.zeros(val.shape)

        for epo in range(self.n_epoch):
            e=0
            h_next = np.zeros((self.m, 1))
            print('\n--- Epoch: ', epo,'--- \n')
            while(e <= len(self.book_data) - self.seq_length-1):
                
                # compute gradients on next sequence of input data
                X, Y = self.convert_to_onehot(self.book_data[e:e+self.seq_length]), self.convert_to_onehot(self.book_data[e+1:e+self.seq_length+1]) # shapes (d, seq_len) = (80 x 25) and (K, seq_len) = (80 x 25)            
                gradients, h_next, loss = self.compute_gradients(X, Y, h_next)
                e += self.seq_length

                # compute smooth loss
                if t_step == 0:                     
                    smooth_loss =  loss
                else:
                    smooth_loss =  0.999 * smooth_loss + 0.001 * loss
                
                # log smallest loss found
                if smooth_loss < min_loss[1]:
                    min_loss = (t_step,smooth_loss)
                    self.best_net_params = self.net_params

                # update parameters
                for key, param in self.net_params.items():
                    sum_of_squared_grads[key] += gradients[key] ** 2
                    self.net_params[key] -= (self.lr/np.sqrt(sum_of_squared_grads[key]+np.finfo(float).eps)) * gradients[key]
                
                if t_step % 10000 == 0:
                    print("\niter=", t_step, "Smooth loss=", smooth_loss)
                
                if t_step % 10000 == 0 and  t_step <= 100000:
                    _,text = self.synthesize_text(h0 = h_next, x0=X[:,0], seq_len=200)
                    print(text)

                #if t_step % 1000 == 0:
                smooth_losses.append(smooth_loss)

                t_step += 1
        return smooth_losses, min_loss


""" gradient testing method"""
def check_grads():
    rnn = RNN(m=100)
    X_chars, Y_chars = rnn.book_data[:rnn.seq_length], rnn.book_data[1:rnn.seq_length+1]
    X, Y = rnn.convert_to_onehot(X_chars), rnn.convert_to_onehot(Y_chars) # shapes (d, seq_len) = (80 x 25) and (K, seq_len) = (80 x 25)            

    analytical_grads, _, _ = rnn.compute_gradients(X,Y, h0=np.zeros((rnn.m, 1)))
    numeric_grads = rnn.compute_grads_num(X, Y, h=1e-4)

    grad_diffs = PrettyTable(['Grad', 'Diff'])
    # compute relative difference between analytical and numerical gradients
    for key, val in analytical_grads.items():
        diff = np.linalg.norm(analytical_grads[key] - numeric_grads[key]) / max( 1e-6, np.linalg.norm(analytical_grads[key]) + np.linalg.norm(numeric_grads[key]))
        grad_diffs.add_row([key, diff])
    print(grad_diffs)

    for key, val in analytical_grads.items():
        np.testing.assert_almost_equal(np.matrix(analytical_grads[key]), np.matrix(numeric_grads[key]), decimal=8)

    return   



""" Runner """
def main():
    check_grads()

    #train model
    rnn = RNN(m=100, n_epoch=2)
    losses, min_loss = rnn.adagrad()
    plt.plot(losses)
    plt.xlabel('time steps')
    plt.ylabel('Smooth loss')
    plt.show()

    x0 = np.zeros((rnn.d,1))    # dummy starting vector
    x0[1] = 1
    _, generated_text = rnn.synthesize_text(np.zeros((rnn.m,1)), x0, 1000)
    print(generated_text)

    print('smallest loss achieved: ', min_loss[1], 'time step:', min_loss[0], '\n')
    rnn.net_params = rnn.best_net_params # set params back to the parameter values that gave smallest loss

    _, generated_text = rnn.synthesize_text(np.zeros((rnn.m,1)), x0, 1000)
    print(generated_text)


if __name__ == "__main__":
    main()




