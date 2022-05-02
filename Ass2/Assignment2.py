
__author__ = 'Léo Noharet'
__email__ = 'lnoharet@kth.se'


import copy
import math
import random
import scipy
from scipy.io import *
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt


# GLOBAL CONSTANTS
n = None # number of images
d = 3072  # dimension of images
K = 10 # number of classes
m = 50 # number of hidden layer nodes
k = 2  # amount of layers 

eta_min = 1e-5
eta_max = 1e-1
n_step = None #2 * n //n_batch # stepsize (rule of thumb: n_s = q*floor(n/n_batch) where 2 < int q < 8)
#np.random.seed(500)

# reads in data from matlab file: fname, converts it into the wanted format and saves it to new_fname. 
def separate_data(fname, new_fname):
    data_batch = loadmat('../Datasets/cifar-10-batches-mat/' + fname)
    x = np.float64(np.matrix(data_batch['data']).transpose())    # image pixel data (d x n)
    y = np.matrix(np.array(data_batch['labels'])) # labels for each of the n images. An index 0-9
    # create a one-hot representation of Y. 
    y_m = np.zeros((K, x.shape[1]), dtype=np.float64)
    for i in range(x.shape[1]):
        j = y[i][0] # label image i
        y_m[j,i] = 1
    scipy.io.savemat('../Datasets/'+new_fname, dict(data=x, onehot = y_m, labels = y))
    return '../Datasets/'+new_fname


# input training data, outputs mean and std
def get_mean_and_std(trainX):
    mean_trainX  = np.matrix(trainX.mean(1)).transpose()
    std_trainX   = np.matrix(trainX.std(1)).transpose()

    return mean_trainX, std_trainX


def normalize_data(X, mean_X, std_X):
    X = X - numpy.matlib.repmat(mean_X, 1, np.size(X,1) )
    X = np.divide(X, numpy.matlib.repmat(std_X, 1, np.size(X,1) ))

    return X


def pre_process(train, val, test):
    # find mean and std of training data
    mean_trainX, std_trainX = get_mean_and_std(train)

    # Normalize training, validation and test datasets
    train = normalize_data(train, mean_trainX, std_trainX)
    val = normalize_data(val, mean_trainX, std_trainX)
    test = normalize_data(test, mean_trainX, std_trainX)

    return train, val, test


def init_parameters():
    
    W1 = np.random.normal(loc=0, scale=1/math.sqrt(d), size=(m,d))
    W2 = np.random.normal(loc=0, scale=1/math.sqrt(m), size=(K,m))

    b1 = np.zeros((m,1), dtype=np.float64)
    b2 = np.zeros((K,1), dtype=np.float64)

    return [W1, W2], [b1, b2]
    

def softmax(x):
	""" Standard definition of the softmax function """
	return np.exp(x) / np.sum(np.exp(x), axis=0)


def evaluate_classifier(X, W, b):

    W1 = W[0]
    W2 = W[1]
    b1 = b[0]
    b2 = b[1]
    s1 = W1 @ X + b1
    h = np.maximum(0, s1) # ReLu activation values
    s = W2 @ h + b2
    return softmax(s), h 


def compute_cost(X, Y, W, b, lamdba_):

    P, _ = evaluate_classifier(X, W, b)
    sum_lcross = 0
    
    for col_idx in range(X.shape[1]):
        y, p = Y[:,col_idx] , P[:,col_idx]
        sum_lcross += -np.dot(y.T, np.log(p))[0,0]

    sum_square_weights = 0
    for l in range(k):
        sum_square_weights += np.sum(np.square(W[l]), dtype=np.float64)
    
    regularization = lamdba_ * sum_square_weights
    loss = 1/X.shape[1] * sum_lcross 
    cost = loss + regularization

    return cost, loss


def compute_accuracy(X, y, W, b):
    # compute models predictions in vector:
    P, _ = evaluate_classifier(X, W, b)
    amount_of_correct = 0
    #for each image in dataset X:
    for img in range(P.shape[1]):
        # find index of the highest prob of the p vector for each image. 
        prediction = np.where(P[:,img] == max(P[:,img]))[0][0]
        if prediction == y[img][0]:
            amount_of_correct += 1
    return amount_of_correct / y.shape[0]


def compute_gradients(X_batch, Y_batch, W, b, lambda_):
    n_b = X_batch.shape[1]
    #forward pass
    H_batch = [ X_batch ]
    for l in range(1, k):
        s = (W[l-1] @ H_batch[l-1]) + np.dot(b[l-1],  np.ones((1,n_b) , dtype=np.float64))
        H_batch.append( np.maximum(s, np.zeros(s.shape, dtype=np.float64)) )
    P_batch = softmax(W[-1] @ H_batch[-1] + np.dot(b[-1], np.ones((1,n_b) , dtype=np.float64)) )

    #backward pass
    G_batch = -(Y_batch - P_batch)

    W_grads = [None] * k
    b_grads = [None] * k

    for l in range(k, 1, -1): 
        W_grads[l-1] = 1/n_b * (G_batch @ H_batch[l-1].T) + 2 * lambda_ * W[l-1]
        b_grads[l-1] = 1/n_b * (G_batch @ np.ones((n_b,1) , dtype=np.float64)) 

        G_batch = W[l-1].T @ G_batch
        G_batch = np.multiply(G_batch, np.where(H_batch[l-1] > 0, 1, 0))

    W_grads[0] = 1/n_b * (G_batch @ X_batch.T) + 2*lambda_ * W[0] 
    b_grads[0] = 1/n_b * (G_batch @ np.ones((n_b,1) , dtype=np.float64)) 
    
    return W_grads, b_grads


def compute_grads_num_slow(X, Y, W, b, lamda, h):
    """ Converted from matlab code """

    grad_W = [None] * k 
    grad_b = [None] * k
    
    for l in range(k):
        grad_b[l] = np.zeros(b[l].shape , dtype=np.float64)

        for i in range(len(b[l])):
            b_try = copy.deepcopy(b)
            b_try[l][i] -= h
            c1, _ = compute_cost(X, Y, W, b_try, lamda)

            b_try = copy.deepcopy(b)
            b_try[l][i] += h
            c2, _ = compute_cost(X, Y, W, b_try, lamda)

            grad_b[l][i] = (c2-c1) / (2*h)

        grad_W[l] = np.zeros(W[l].shape , dtype=np.float64)
        for i in np.ndindex(W[l].shape):
            W_try = copy.deepcopy(W)
            W_try[l][i] -= h
            c1, _ = compute_cost(X, Y, W_try, b, lamda)

            W_try = copy.deepcopy(W)
            W_try[l][i] += h
            c2, _ = compute_cost(X, Y, W_try, b, lamda)

            grad_W[l][i] = (c2-c1) / (2*h)

    return grad_W, grad_b


def mini_batch_GD(train_X, train_Y, train_y, val_X, val_Y, val_y, W, b, GDparams, lamb_srch, rand_batch):

    lambd   = GDparams[0]
    n_batch = GDparams[1]
    cycles  = GDparams[2]

    tot_t = cycles * 2 * n_step # tot number of updates of eta
    n_epoch = tot_t // (train_X.shape[1] // n_batch) # num of epochs to fit the requested amount of cycles

    etas              = []
    val_accuracies    = []
    train_accuracies  = []
    training_costs    = []
    training_losses   = []
    validation_costs  = []
    validation_losses = []

    t = 0
    eta_t = eta_min
    for i in range(n_epoch):
        if rand_batch:
            r_idx = np.random.permutation(train_X.shape[1])
        for j in range(train_X.shape[1] // n_batch):
            start = j * n_batch
            end = (j+1) * n_batch
            if rand_batch:
                rand_range = r_idx[range(start,end)]
                X_batch = train_X[:, rand_range] 
                Y_batch = train_Y[:, rand_range] 
            else:
                X_batch = train_X[:, start:end] 
                Y_batch = train_Y[:, start:end] 

            W_grads, b_grads = compute_gradients(X_batch, Y_batch, W, b ,lambd)
            for layer in range(k):
                W[layer] -= eta_t * W_grads[layer]
                b[layer] -= eta_t * b_grads[layer]
            # update learning rate cyclically [implementation from Leslie Smith, 2015]
            t+=1
            cycle = math.floor(t/(2*n_step)) # num of cycles done
            etas.append(eta_t)
     
            if (t <= (2 * cycle + 1) * n_step) and (t >=2 * cycle * n_step):
                eta_t = eta_min+((t - (2 * cycle * n_step)) / n_step)*(eta_max - eta_min)
            elif (t <= 2 * (cycle + 1) * n_step) and (t >= (2 * cycle + 1) * n_step) :
                eta_t = eta_max-((t - ((2 * cycle + 1) * n_step)) / n_step)*(eta_max - eta_min)
            
            if not lamb_srch:
                # Compute loss, cost, accuracy for each iteration and store in arrays
                if( t % ((2*n_step)/10) == 0 ):
                    train_acc = compute_accuracy(train_X, train_y, W,b)
                    val_acc   = compute_accuracy(val_X, val_y, W,b)
                    train_accuracies.append( train_acc )
                    val_accuracies.append( val_acc )
                    train_cost, train_loss = compute_cost (train_X, train_Y, W, b, lambd)
                    training_costs.append( train_cost )
                    training_losses.append( train_loss )

                    val_cost, val_loss = compute_cost (val_X, val_Y, W, b, lambd)
                    validation_costs.append( val_cost )
                    validation_losses.append( val_loss )
            
    return W, b, training_costs, training_losses, validation_costs, validation_losses, etas, train_accuracies, val_accuracies


def plot_res(train_costs, val_costs, train_losses, val_losses,train_accuracies, val_accuracies, fig_idx):

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12,3))

    ax1.plot(range(len(val_costs)), val_costs, label= 'Validation', color= 'Red')
    ax1.plot(range(len(train_costs)), train_costs, label= 'Training', color= 'Green')
    ax1.set_title('Cost plot')
    ax1.set(xlabel='Epochs', ylabel='Cost')

    ax2.plot(range(len(val_losses)), val_losses, label= 'Validation ', color= 'Red')
    ax2.plot(range(len(train_losses)), train_losses, label= 'Training', color= 'Green')
    ax2.set_title('Loss plot')
    ax2.set(xlabel='Epochs', ylabel='Loss')

    ax3.plot(range(len(val_accuracies)), val_accuracies, label= 'Validation', color= 'Red')
    ax3.plot(range(len(train_accuracies)), train_accuracies, label= 'Training', color= 'Green')
    ax3.set_title('Accuracy plot')
    ax3.set(xlabel='Epochs', ylabel='Accuracy')

    fig.tight_layout()
    plt.legend()
    #plt.show()
    fig.savefig('Result_Pics/Figure_' + str(fig_idx) + '.png', bbox_inches='tight')


def plot_learning_rates(etas, cycles):

    plt.plot(range(len(etas)), etas, label= 'Cyclic learning rate' , color= 'Blue')
    plt.plot(range(len(etas)), [eta_max for i in range(len(etas))], label= 'eta_max' , color= 'Red')
    plt.plot(range(len(etas)), [eta_min for i in range(len(etas))], label= 'eta_min' , color= 'Green')
    plt.xlabel("Update steps")
    plt.ylabel('eta_t')
    plt.legend()
    #plt.show()
    plt.savefig('Result_Pics/' + 'cycles_' + str(cycles) + '.png', bbox_inches='tight')
    plt.close() 


def lamdba_search():
    # load data
    training1 = loadmat(separate_data('data_batch_1.mat', 'training_batch1.mat'))
    training2 = loadmat(separate_data('data_batch_2.mat', 'training_batch2.mat'))
    training3 = loadmat(separate_data('data_batch_3.mat', 'training_batch3.mat'))
    training4 = loadmat(separate_data('data_batch_4.mat', 'training_batch4.mat'))
    training5 = loadmat(separate_data('data_batch_5.mat', 'training_batch5.mat'))
    test_batch = loadmat(separate_data('test_batch.mat', 'testing_batch.mat'))

    train_X = np.concatenate(( training1['data'], training2['data'], training3['data'], training4['data'], training5['data']), axis=1)
    train_Y = np.concatenate(( training1['onehot'], training2['onehot'], training3['onehot'], training4['onehot'], training5['onehot']), axis=1)
    train_y = np.concatenate(( training1['labels'], training2['labels'], training3['labels'], training4['labels'], training5['labels']), axis=0)
    
    val_X = train_X[:, -5000:]
    val_Y = train_Y[:, -5000:]
    val_y = train_y[-5000:, :]
    train_X = train_X[:, :-5000]
    train_Y = train_Y[:, :-5000]
    train_y = train_y[:-5000, :]
    test_X = test_batch['data']
    train_X, val_X, test_X = pre_process(train_X, val_X, test_X)

    ## COARSE SEARCH
    #           lambda, batchsize, cycles
    GDparams = [ 0,     100,       3]
    global n, n_step 
    n = train_X.shape[1]
    n_step = 2* n//GDparams[1]

    lamdas = np.arange(-5, -1, 0.5)
    val_accuracies = []
    for lamb in lamdas:
        W, b = init_parameters()
        GDparams[0] = 10**lamb
        W, b, _, _, _, _, _, _, _ = mini_batch_GD(train_X, train_Y, train_y, val_X, val_Y, val_y, W, b, GDparams, lamb_srch=True, rand_batch=False)
        val_accuracies.append( compute_accuracy(val_X, val_y, W, b) )
        
    #Plot results
    plt.scatter([10**q for q in lamdas], val_accuracies)
    plt.xlabel('lambda')
    plt.ylabel('val accuracy')
    plt.savefig('Result_Pics/coarse_seach5.png')
    plt.close()
    print(val_accuracies)
    print([10**q for q in lamdas])

    ### FINE RANDOM SEARCH
    best_3_lamb = np.take(lamdas, np.argsort(val_accuracies)[-3:])

    l_min, l_max = min(best_3_lamb), max(best_3_lamb)
    accs = []
    lambs = []
    for _ in range(20):
        W, b = init_parameters()
        l = l_min + (l_max - l_min) * random.uniform(0,1)
        lambs.append(l)
        GDparams[0] = 10**l
        W, b, _, _, _, _, _, _, _ = mini_batch_GD(train_X, train_Y, train_y, val_X, val_Y, val_y, W, b, GDparams, lamb_srch=True, rand_batch=False)
        accs.append( compute_accuracy(val_X, val_y, W, b) )

    print(accs)
    print(lambs)
    best_3_lamb = np.take(lambs, np.argsort(accs)[-3:])
    print("The 3 best lambda values:", best_3_lamb, "and their respective accuracies on validation dataset",np.take(accs, np.argsort(accs)[-3:]))

    #Plot results
    plt.scatter([10**q for q in lambs], accs)
    plt.xlabel('lambda')
    plt.ylabel('val accuracy')
    plt.savefig('Result_Pics/fine_seach5.png')
    plt.close()

    return np.take(lambs, np.argsort(accs)[-1:])

def train_best_network(lamb):
    # load data
    training1 = loadmat(separate_data('data_batch_1.mat', 'training_batch1.mat'))
    training2 = loadmat(separate_data('data_batch_2.mat', 'training_batch2.mat'))
    training3 = loadmat(separate_data('data_batch_3.mat', 'training_batch3.mat'))
    training4 = loadmat(separate_data('data_batch_4.mat', 'training_batch4.mat'))
    training5 = loadmat(separate_data('data_batch_5.mat', 'training_batch5.mat'))
    test_batch = loadmat(separate_data('test_batch.mat', 'testing_batch.mat'))
    train_X = np.concatenate(( training1['data'], training2['data'], training3['data'], training4['data'], training5['data']), axis=1)
    train_Y = np.concatenate(( training1['onehot'], training2['onehot'], training3['onehot'], training4['onehot'], training5['onehot']), axis=1)
    train_y = np.concatenate(( training1['labels'], training2['labels'], training3['labels'], training4['labels'], training5['labels']), axis=0)
    
    val_X = train_X[:, -1000:]
    val_Y = train_Y[:, -1000:]
    val_y = train_y[-1000:, :]
    train_X = train_X[:, :-1000]
    train_Y = train_Y[:, :-1000]
    train_y = train_y[:-1000, :]

    test_X = test_batch['data']
    test_y = test_batch['labels']

    train_X, val_X, test_X = pre_process(train_X, val_X, test_X)
    GDparams = [ lamb,     100,       3]
    global n, n_step 
    n = train_X.shape[1]
    n_step = 2* n//GDparams[1]
    W, b = init_parameters()
    W, b, train_costs, train_losses, val_costs, val_losses, etas, train_accuracies, val_accuracies = mini_batch_GD(train_X, train_Y, train_y, val_X, val_Y, val_y, W, b, GDparams, lamb_srch = False, rand_batch = False)
    plot_res(train_costs, val_costs, train_losses, val_losses,train_accuracies, val_accuracies, 5)
    
    # test the accuracy of model on test data
    acc = compute_accuracy(test_X, test_y, W, b) * 100
    print("Accuracy of testdata = " + str(acc) + "%")
    return acc

def main():
    # Coarse and fine search giving the best lambda
    lmda = lamdba_search()[0]
    print(lmda)

    #Training all of the data with the best found lmda value
    train_best_network(10**lmda)

    #####
    training_path    = separate_data('data_batch_1.mat', 'training_batch.mat')
    validation_path  = separate_data('data_batch_2.mat', 'validation_batch.mat')
    test_path        = separate_data('test_batch.mat', 'testing_batch.mat')
    training_batch   = loadmat(training_path)
    validation_batch = loadmat(validation_path)
    test_batch       = loadmat(test_path)

    training_batch['data'], validation_batch['data'], test_batch['data'] = pre_process(training_batch['data'], validation_batch['data'], test_batch['data'])
  
    train_X = np.matrix(training_batch['data'])
    train_Y = np.matrix(training_batch['onehot'])
    train_y = np.matrix(training_batch['labels'])

    val_X = np.matrix(validation_batch['data'])
    val_Y = np.matrix(validation_batch['onehot'])
    val_y = np.matrix(validation_batch['labels'])

    test_X = np.matrix(test_batch['data'])
    test_y = np.matrix(test_batch['labels'])
    
    #          [ lamdba, n_batch, cycles ]
    GDparams = [ 0.01,   100,     3]
    global n, n_step
    n = train_X.shape[1]
    n_step = 2* n//GDparams[1]

    W, b = init_parameters()

    
    # TESTING CORRECTNESS OF GRADIENTS
    W_grads, b_grads = compute_gradients(train_X[:, :100], train_Y[:, :100], W, b, 0.01)
    W_grads_num, b_grads_num = compute_grads_num_slow(train_X[:, :100], train_Y[:, :100], W, b, 0.01, 1e-5)
    
    for layer in range(k):
        # Test that numerical and analytical gradient computations are the same
        diff_W = np.linalg.norm((W_grads[layer]) - np.matrix(W_grads_num[layer])) / max( 1e-6, np.linalg.norm((W_grads[layer])) + np.linalg.norm((W_grads_num[layer])))
        diff_b = np.linalg.norm((b_grads[layer]) - np.matrix(b_grads_num[layer])) / max( 1e-6, np.linalg.norm((b_grads[layer])) + np.linalg.norm((b_grads_num[layer])))
        print("diff_W = ", diff_W)
        print("diff_b = ", diff_b)
        np.testing.assert_almost_equal(np.matrix(W_grads[layer]), np.matrix(W_grads_num[layer]), decimal=8)
        np.testing.assert_almost_equal(np.matrix(b_grads[layer]), np.matrix(b_grads_num[layer]), decimal=8)
    # perform GD on training data.
    W, b, train_costs, train_losses, val_costs, val_losses, etas, train_accuracies, val_accuracies = mini_batch_GD(train_X, train_Y, train_y, val_X, val_Y, val_y, W, b, GDparams, lamb_srch = False, rand_batch = False)
    
    plot_learning_rates(etas, GDparams[2])
    # plot weight, loss and cost to mimic figures 3 and 4 of assignment
    plot_res(train_costs, val_costs, train_losses, val_losses,train_accuracies, val_accuracies, 4)

    # test the accuracy of model on test dataset
    acc = compute_accuracy(test_X, test_y, W, b) * 100
    print("Accuracy of testdata = " + str(acc) + "%")



if __name__ == "__main__":
    main()

