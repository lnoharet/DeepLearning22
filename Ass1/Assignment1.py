import scipy
from scipy.io import *
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt


__author__ = 'Léo Noharet'
__email__ = 'lnoharet@kth.se'



# GLOBAL CONSTANTS
n = 10000
d = 3072
K = 10

np.random.seed(400)

# reads in data from matlab file: fname, converts it into the wanted format and saves it to new_fname. 
def separate_data(fname, new_fname):
    data_batch = loadmat('../Datasets/cifar-10-batches-mat/' + fname)

    # image pixel data (d x n). Should be double and values between 0 and 1 ??
    x = np.float64(np.matrix(data_batch['data']).transpose())

    y = np.matrix(np.array(data_batch['labels'])) # labels for each of the n images. An index 0-9

    # create a one-hot representation of Y. 
    y_m = np.zeros((K, n), dtype=np.float64)
    for i in range(n):
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
    test   = normalize_data(test, mean_trainX, std_trainX)

    return train, val, test


def init_parameters():
    W = np.random.normal(loc=0, scale=0.01, size=(K,d))
    b = np.random.normal(loc=0, scale=0.01, size=(K,1))
    return W, b
    

def softmax(x):
	""" Standard definition of the softmax function """
	return np.exp(x) / np.sum(np.exp(x), axis=0)

def evaluate_classifier(X, weight, bias):
    return softmax(weight @ X + bias)

def compute_cost(X, Y, W, b, lamdba_):

    P = evaluate_classifier(X, W, b)
    sum_lcross = 0
    
    for col_idx in range(X.shape[1]):
        y, p = Y[:,col_idx] , P[:,col_idx]
        sum_lcross += -np.dot(y.T, np.log(p))[0,0]

    regularization = lamdba_ * np.sum(np.square(W))
    loss = 1/X.shape[1] * sum_lcross 
    cost = loss + regularization

    return cost, loss


def compute_accuracy(X, y, W, b):
    # compute models predictions in vector:
    P = evaluate_classifier(X, W, b)
    amount_of_correct = 0
    #for each image in dataset X:
    for img in range(P.shape[1]):
        # find index of the highest prob of the p vector for each image. 
        prediction = np.where(P[:,img] == max(P[:,img]))[0][0]
        if prediction == y[img][0]:
            amount_of_correct += 1

    return amount_of_correct / y.shape[0]


def compute_gradients(X_batch, Y_batch, P, W, lambda_, b):
    n = X_batch.shape[1]
    #forward pass
    P_batch = evaluate_classifier(X_batch, W, b)
    
    #backward pass
    G_batch = -(Y_batch - P_batch)

    grad_W = 1/n * np.matmul(G_batch, X_batch.T) + 2*lambda_ * W 
    grad_b = 1/n * np.matmul(G_batch, np.ones((n,1) , dtype=np.float64)) #.reshape(n, 1))


    return grad_W, grad_b


def compute_grads_num_slow(X, Y, W, b, lamda, h):
	""" Converted from matlab code """
	no 	= 	W.shape[0]
	d 	= 	X.shape[0]

	grad_W = np.zeros(W.shape , dtype=np.float64)
	grad_b = np.zeros((no, 1) , dtype=np.float64)
	
	for i in range(len(b)):
		b_try = np.array(b)
		b_try[i] -= h
		c1, _ = compute_cost(X, Y, W, b_try, lamda)

		b_try = np.array(b)
		b_try[i] += h
		c2, _ = compute_cost(X, Y, W, b_try, lamda)

		grad_b[i] = (c2-c1) / (2*h)

	for i in range(W.shape[0]):
		for j in range(W.shape[1]):
			W_try = np.array(W)
			W_try[i,j] -= h
			c1, _ = compute_cost(X, Y, W_try, b, lamda)

			W_try = np.array(W)
			W_try[i,j] += h
			c2, _ = compute_cost(X, Y, W_try, b, lamda)

			grad_W[i,j] = (c2-c1) / (2*h)

	return grad_W, grad_b


def mini_batch_GD(train_X, train_Y, val_X, val_Y, W, b, GDparams):

    lambd   = GDparams[0]
    n_epoch = GDparams[1] 
    n_batch = GDparams[2]
    eta     = GDparams[3]

    training_costs    = []
    training_losses   = []
    validation_costs  = []
    validation_losses = []

    for i in range(n_epoch):
        for j in range(n // n_batch):
            start = j * n_batch
            end = (j+1) * n_batch
            X_batch = train_X[:, start:end]
            Y_batch = train_Y[:, start:end]
            P = evaluate_classifier(X_batch, W, b)
            grad_W, grad_b = compute_gradients(X_batch, Y_batch, P, W, lambd, b)
            W -= eta * grad_W
            b -= eta * grad_b

        ## Compute loss and cost for each iteration and store in arrays
        train_cost, train_loss = compute_cost (train_X, train_Y, W, b, lambd)
        training_costs.append(train_cost)
        training_losses.append(train_loss)

        val_cost, val_loss = compute_cost (val_X, val_Y, W, b, lambd)
        validation_costs.append(val_cost)
        validation_losses.append(val_loss)


    return W, b, training_costs, training_losses, validation_costs, validation_losses


# Given function from canvas to display weight matrix W
def montage(W, param_idx):
    """ Display the image for each label in W """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2,5)
    for i in range(2):
        for j in range(5):
            im  = W[i*5+j,:].reshape(32,32,3, order='F')
            sim = (im-np.min(im[:]))/(np.max(im[:])-np.min(im[:]))
            sim = sim.transpose(1,0,2)
            ax[i][j].imshow(sim, interpolation='nearest')
            ax[i][j].set_title("y="+str(5*i+j))
            ax[i][j].axis('off')
    #plt.show()
    fig.savefig('Result_Pics/' + 'weights_params' + str(param_idx+1) + '.png', bbox_inches='tight')
    plt.close(fig) 

def plot_res(train, val, type, param_idx):

    plt.plot(range(len(val)), val, label= 'Validation ' + type, color= 'Red')
    plt.plot(range(len(train)), train, label= 'Training ' + type, color= 'Green')

    plt.xlabel("Epochs")
    plt.ylabel(type)

    plt.legend()
    #plt.show()
    plt.savefig('Result_Pics/' + type + '_for_' + str(param_idx+1) + '.png', bbox_inches='tight')
    plt.close() 




def main():

    training_path    = separate_data('data_batch_1.mat', 'training_batch.mat')
    validation_path  = separate_data('data_batch_2.mat', 'validation_batch.mat')
    test_path        = separate_data('test_batch.mat', 'testing_batch.mat')
    training_batch   = loadmat(training_path)
    validation_batch = loadmat(validation_path)
    test_batch       = loadmat(test_path)

    training_batch['data'], validation_batch['data'], test_batch['data'] = pre_process(training_batch['data'], validation_batch['data'], test_batch['data'])
  
    train_X = np.matrix(training_batch['data'])
    train_Y = np.matrix(training_batch['onehot'])

    val_X = np.matrix(validation_batch['data'])
    val_Y = np.matrix(validation_batch['onehot'])

    test_X = np.matrix(test_batch['data'])
    test_y = np.matrix(test_batch['labels'])
    
    
    #            [ lamdba, n_epoch, n_batch, eta ]
    GDparams = [ [ 0,      40,      100,     0.1 ], [0, 40, 100, 0.001], [0.1, 40, 100, 0.001], [1, 40, 100, 0.001]  ]

    for i in range(len(GDparams)):
        W, b = init_parameters()
        lambd   = GDparams[i][0]
        n_epoch = GDparams[i][1] 
        n_batch = GDparams[i][2]
        eta     = GDparams[i][3]

        P = evaluate_classifier(train_X, W, b)
        
        grad_W, grad_b = compute_gradients(train_X[:, :n_batch], train_Y[:, :n_batch], P, W, lambd, b)
        ngrad_W, ngrad_b = compute_grads_num_slow(train_X[:, :n_batch], train_Y[:, :n_batch], W, b, lambd, 1e-6)

        # Test that numerical and analytical gradient computations are the same
        diff_W = np.linalg.norm((grad_W) - np.matrix(ngrad_W)) / max( 1e-6, np.linalg.norm((grad_W)) + np.linalg.norm((ngrad_W)))
        diff_b = np.linalg.norm((grad_b) - np.matrix(ngrad_b)) / max( 1e-6, np.linalg.norm((grad_b)) + np.linalg.norm((ngrad_b)))
        print("diff_W = ", diff_W)
        print("diff_b = ", diff_b)
        np.testing.assert_almost_equal(np.matrix(grad_W), np.matrix(ngrad_W), decimal=8)
        np.testing.assert_almost_equal(np.matrix(grad_b), np.matrix(ngrad_b), decimal=8)
        
        # perform GD on training data.
        W, b, train_costs, train_losses, val_costs, val_losses = mini_batch_GD(train_X, train_Y, val_X, val_Y, W, b, GDparams[i])

        # plot weight, loss and cost
        montage(W, i)
        plot_res(train_costs, val_costs, "costs", i )
        plot_res(train_losses, val_losses, "losses", i )

        # test the accuracy of model
        acc = compute_accuracy(test_X, test_y, W, b) * 100
        print("Accuracy of testdata = " + str(acc) + "%")

    

if __name__ == "__main__":
    main()