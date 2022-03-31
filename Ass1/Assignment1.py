import math
from tokenize import Double
import scipy
from functions import *
from scipy.io import *
import numpy as np
import numpy.matlib

# GLOBAL CONSTANTS
n = 10000
d = 3072
K = 10

test_batch_ = loadmat('Datasets/cifar-10-batches-mat/test_batch.mat')     # for testing
data_batch_1 = loadmat('Datasets/cifar-10-batches-mat/data_batch_1.mat') # for training
data_batch_2 = loadmat('Datasets/cifar-10-batches-mat/data_batch_2.mat') # for validation

data_batches = ['data_batch_3.mat', 'data_batch_4.mat','data_batch_5.mat']
# keys = ['__header__', '__version__', '__globals__', 'data', 'labels', 'batch_label']

test_batch = None
training_batch = None
validation_batch = None

#GLOBAL PARAMETERS
W = None # Weights matrix
b = None # bias vector

def separate_data(fname, new_fname):
    data_batch = loadmat('Datasets/cifar-10-batches-mat/' + fname)

    # image pixel data (d x n). Should be double and values between 0 and 1 ??
    x = np.matrix(np.array(data_batch['data']).transpose()).astype(float)
    #print(X)

    y = np.matrix(np.array(data_batch['labels'])) # labels for each of the n images. An index 0-9
    #print(y)

    y_m = np.zeros((K, n))
    for i in range(n):
        j = y[i][0] # label image i
        y_m[j,i] = 1

    return save_data_to_file(new_fname, x, y_m, y)
     
def save_data_to_file(new_fname, X, Y, y):
    scipy.io.savemat('Datasets/'+new_fname, dict(data=X, onehot = Y, labels = y))
    return 'Datasets'+new_fname

# input training data, outputs mean and std
def get_mean_and_std(trainX):
    mean_trainX  = np.matrix(trainX.mean(1)).transpose()
    std_trainX   = np.matrix(trainX.std(1)).transpose()

    print('shape of mean_trainX ', mean_trainX.shape)
    print('shape of std_trainX ', std_trainX.shape)

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
    global W,b
    W = np.random.normal(loc=0, scale=0.01, size=(K,d))
    b = np.random.normal(loc=0, scale=0.01, size=(K,1))
    
def EvaluateClassifier(X, weight, bias):
    P = np.zeros((K,n))

    for col_idx in range(X.shape[1]):
        col = X[:, col_idx]
        s = numpy.matmul(weight, col) + bias

        p = softmax(s)
        P[:,col_idx] = p.T

    return P

def sum_square_elem_of_m(m):
    rows, cols = m.shape
    tot_sum = 0
    row = 0
    while row < rows:
        col = 0
        while col < cols:
            tot_sum += m[row, col] * m[row, col]
            col += 1
        row += 1
    return tot_sum

def cross_entropy_loss(x, y, W, b, p):
    return math.log(p[y])

def ComputeCost(X, Y, W, b, lamdba_):
    size_D = X.size
    W_squared_sum = sum_square_elem_of_m(W)
    P = EvaluateClassifier(X, W, b)
    sum_lcross = 0
    for col_idx in range(X.shape[1]):
        # choose one image vector in X and P
        x, p = X[:,col_idx] , P[:,col_idx]
        # find corresponding label in Y matrix
        y =  np.where(Y[:,col_idx] == 1)
        if y[0].size > 1:
            print("oh oh, several labels for this image")
        else:
            sum_lcross += cross_entropy_loss(x, y[0], W, b, p)
    return 1/size_D * sum_lcross + lamdba_ * W_squared_sum

def ComputeAccuracy(X, y, W, b):
    # compute models predictions in vector:
    P = EvaluateClassifier(X, W, b)
    amount_of_correct = 0
    #for each image in dataset X:
    for img in range(P.shape[1]):
        # find index of the max value of the p vector for each image. 
        prediction = np.where(P[:,img] == max(P[:,img]))[0][0]
        print("y[img] ", y[img][0])
        print("prediction ",prediction, "\n")
        if prediction == y[img][0]:
            amount_of_correct += 1
    print(amount_of_correct, "/ ", y.shape[0])
    return amount_of_correct / y.shape[0]

def main():
    init_parameters()

    training_path    = separate_data('data_batch_1.mat', 'training_batch.mat')
    validation_path  = separate_data('data_batch_2.mat', 'validation_batch.mat')
    test_path        = separate_data('test_batch.mat', 'testing_batch.mat')

    training_batch   = loadmat(training_path)
    validation_batch = loadmat(validation_path)
    test_batch       = loadmat(test_path)

    training_batch['data'], validation_batch['data'], test_batch['data'] = pre_process(training_batch['data'], validation_batch['data'], test_batch['data'])
    trainX = training_batch['data']
    trainY = training_batch['onehot']
    trainy = training_batch['labels']
    #print(training_batch['data'])
    temp4 = EvaluateClassifier(trainX[:,:100], W, b)

    temp5 = ComputeCost(trainX, trainY, W, b, 1)
    temp6 = ComputeAccuracy(trainX, trainy, W, b)

    print(temp6)
main()
