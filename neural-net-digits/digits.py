"""digits.py implements 2-layer and multi-layer neural network machine learning
to classify images of handwritten digits."""

__authors__  = "Wei Zhen Teoh, Ruiwen Li"

from pylab import *
from numpy import *
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import matplotlib.patches as mpatches
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random

import cPickle

import os
from scipy.io import loadmat

#Load the MNIST digit data
M = loadmat("mnist_all.mat")

snapshot = cPickle.load(open("snapshot50.pkl"))
W0sample = snapshot["W0"]
W1sample = snapshot["W1"]
Wsample = dot(W0sample,W1sample)
b0sample = snapshot["b0"].reshape((300,1))
b1sample = snapshot["b1"].reshape((10,1))
bsample = b1sample


#part1

#Collect 10 images of each digit from the training set
f, axarr = plt.subplots(10, 10)
for i in range(10):
    for j in range(10):
        fig= axarr[i, j].imshow(M["train"+str(i)][j].reshape((28,28)),\
                                cmap=cm.gray)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
f.savefig('part1.png')
plt.close("all")



#Part 2
def linComb(x, W, b):
    return dot(W.T,x)+b
    #W is weight matrix of dimension JxI, each entry is w_ji, 
    #b is bias and has dimension Ix1
    
    
    
    
#Part 3    


def softmax(y):
    '''Return the output of the softmax function for the matrix of output y. y
    is an NxM matrix where N is the number of outputs for a single case, and M
    is the number of cases'''
    return exp(y)/tile(sum(exp(y),0), (len(y),1))

def cost(y, y_):
    '''Return the sum of negative log-probabilities of the correct answer for 
    the M cases. y_ is the NxM one-hot encoded matrix and y is NxM matrix 
    of the probabilities of the N outputs of all M cases.'''
    return -sum(y_*log(y)) 
#one-hot encoded matrix is one which has value 1 at entry corresponding
#to the correct digit for each case
#All the other entries have value 0

def deriv_2layer(x, W, b, y_):
    '''Return the derivative of the cost function with respect to each entry in 
    the weight matrix W and the bias b. x is 784xM input matrix, M is the 
    number of cases. y_ is the one-hot encoded matrix.'''
    L = linComb(x, W, b)
    y = softmax(L)
    dCdL = y-y_
    dCdW = dot(x, dCdL.T ) 
    dCdb = dot(ones(x.shape[1]), dCdL.T).reshape(b.shape[0],1)
    return dCdW, dCdb
    
    
    
#Part 4

def appderiv_2layer(x, W, b, y_, d):
    '''Return the derivatives of the cost function with respect to the weight 
    matrix W and bias b using finite difference approximation. x is 784xM input 
    matrix, M is the number of cases. y_ is the one-hot encoded matrix. d is the
    size of run in the gradient approximation.'''
    y = softmax(linComb(x, W, b))
    fixpoint = cost(y, y_)
    
    dCdW = zeros(W.shape)
    dCdb = zeros(W.shape[1])
    for i in range(W.shape[1]):
        for j in range(W.shape[0]):
            Wd = copy(W).astype(float)
            Wd[j][i] += d
            y = softmax(linComb(x, Wd, b))
            dCdW[j][i] = (cost(y,y_)-fixpoint)/d
    for i in range(W.shape[1]):
        bd = copy(b).astype(float)
        bd[i] += d
        y = softmax(linComb(x, W, bd))
        dCdb[i] = (cost(y,y_)-fixpoint)/d
    return dCdW, dCdb.reshape(dCdb.shape[0], 1)

print 'Verifying the gradient calculation using random coods of small dimension'
random.seed(0)
for n in range(3):
    W = random.rand(3,2)
    b = random.rand(2).reshape(2, 1)
    x = random.rand(3,3)
    y_ = array([[0, 0, 1], [1, 1, 0]])
    print "the actual gradient calculated"
    dCdW, dCdb = deriv_2layer(x, W, b, y_)
    print "dCdW = "+ str(dCdW) 
    print "dCdb = " + str(dCdb)
    print "the gradient obtained by finite-difference approximation"
    dCdW, dCdb = appderiv_2layer(x, W, b, y_, 0.01)
    print "dCdW = "+ str(dCdW)
    print "dCdb = " + str(dCdb)



#Part 5
#setting up trainset
trainset = (vstack((M['train0'], M['train1'], M['train2'], M['train3'],\
                    M['train4'], M['train5'], M['train6'], M['train7'],\
                    M['train8'], M['train9'])).T)/255.
                    
trainlist0 = tile(array([0]),len(M['train0']))
trainlist1 = tile(array([1]),len(M['train1']))
trainlist2 = tile(array([2]),len(M['train2']))
trainlist3 = tile(array([3]),len(M['train3']))
trainlist4 = tile(array([4]),len(M['train4']))
trainlist5 = tile(array([5]),len(M['train5']))
trainlist6 = tile(array([6]),len(M['train6']))
trainlist7 = tile(array([7]),len(M['train7']))
trainlist8 = tile(array([8]),len(M['train8']))
trainlist9 = tile(array([9]),len(M['train9']))
traintruth = concatenate((trainlist0, trainlist1, trainlist2, trainlist3, \
                        trainlist4, trainlist5, trainlist6, trainlist7, \
                        trainlist8, trainlist9))

trainvec0 = tile(array([1,0,0,0,0,0,0,0,0,0]), (len(M['train0']),1))
trainvec1 = tile(array([0,1,0,0,0,0,0,0,0,0]), (len(M['train1']),1))
trainvec2 = tile(array([0,0,1,0,0,0,0,0,0,0]), (len(M['train2']),1))
trainvec3 = tile(array([0,0,0,1,0,0,0,0,0,0]), (len(M['train3']),1))
trainvec4 = tile(array([0,0,0,0,1,0,0,0,0,0]), (len(M['train4']),1))
trainvec5 = tile(array([0,0,0,0,0,1,0,0,0,0]), (len(M['train5']),1))
trainvec6 = tile(array([0,0,0,0,0,0,1,0,0,0]), (len(M['train6']),1))
trainvec7 = tile(array([0,0,0,0,0,0,0,1,0,0]), (len(M['train7']),1))
trainvec8 = tile(array([0,0,0,0,0,0,0,0,1,0]), (len(M['train8']),1))
trainvec9 = tile(array([0,0,0,0,0,0,0,0,0,1]), (len(M['train9']),1))
trainmatrix = vstack((trainvec0, trainvec1, trainvec2, trainvec3, trainvec4, \
                    trainvec5, trainvec6, trainvec7, trainvec8, trainvec9)).T

#setting up testset
testset = (vstack((M['test0'], M['test1'], M['test2'], M['test3'], M['test4'], \
            M['test5'], M['test6'], M['test7'], M['test8'], M['test9'])).T)/255.

testlist0 = tile(array([0]),len(M['test0']))
testlist1 = tile(array([1]),len(M['test1']))
testlist2 = tile(array([2]),len(M['test2']))
testlist3 = tile(array([3]),len(M['test3']))
testlist4 = tile(array([4]),len(M['test4']))
testlist5 = tile(array([5]),len(M['test5']))
testlist6 = tile(array([6]),len(M['test6']))
testlist7 = tile(array([7]),len(M['test7']))
testlist8 = tile(array([8]),len(M['test8']))
testlist9 = tile(array([9]),len(M['test9']))
testtruth = concatenate((testlist0, testlist1, testlist2, testlist3, testlist4,\
                        testlist5, testlist6, testlist7, testlist8, testlist9))

testvec0 = tile(array([1,0,0,0,0,0,0,0,0,0]), (len(M['test0']),1))
testvec1 = tile(array([0,1,0,0,0,0,0,0,0,0]), (len(M['test1']),1))
testvec2 = tile(array([0,0,1,0,0,0,0,0,0,0]), (len(M['test2']),1))
testvec3 = tile(array([0,0,0,1,0,0,0,0,0,0]), (len(M['test3']),1))
testvec4 = tile(array([0,0,0,0,1,0,0,0,0,0]), (len(M['test4']),1))
testvec5 = tile(array([0,0,0,0,0,1,0,0,0,0]), (len(M['test5']),1))
testvec6 = tile(array([0,0,0,0,0,0,1,0,0,0]), (len(M['test6']),1))
testvec7 = tile(array([0,0,0,0,0,0,0,1,0,0]), (len(M['test7']),1))
testvec8 = tile(array([0,0,0,0,0,0,0,0,1,0]), (len(M['test8']),1))
testvec9 = tile(array([0,0,0,0,0,0,0,0,0,1]), (len(M['test9']),1))
testmatrix = vstack((testvec0, testvec1, testvec2, testvec3, testvec4, \
                    testvec5, testvec6, testvec7, testvec8, testvec9)).T


def record(W, b, traincosts, testcosts, trainperformance, testperformance):
    trainprob = softmax(linComb(trainset, W, b))
    trainprediction = trainprob.argmax(axis=0)
    incorrect = count_nonzero(trainprediction - traintruth)
    trainperformance.append((float(trainset.shape[1])-incorrect)*100/(trainset.shape[1]))
    traincosts.append(cost(trainprob,trainmatrix))
    
    testprob = softmax(linComb(testset, W, b))
    testprediction = testprob.argmax(axis=0)
    incorrect = count_nonzero(testprediction - testtruth)
    testperformance.append((float(testset.shape[1])-incorrect)*100/(testset.shape[1]))
    testcosts.append(cost(testprob,testmatrix))


#mini batch gradient descent

alpha = 0.01
n=0
W = Wsample
b = bsample

traincosts1 = []
trainperformance1 = []
testperformance1 = []
testcosts1 = []

random.seed(0)
while n <= 5000:
     
    if n%100 == 0:
        print 'gradient descent iteration ' + str(n) 
        record(W, b, traincosts1, testcosts1, trainperformance1, testperformance1)
        if traincosts1[-1] <= amin(traincosts1):
            bestW = W
            bestb = b
    chosen = random.choice(range(trainset.shape[1]), 50)
    x = (trainset.T[[chosen]]).T
    y_ = (trainmatrix.T[[chosen]]).T   
    dCdW, dCdb = deriv_2layer(x, W, b, y_)
    W = W - alpha*dCdW
    b = b - alpha*dCdb
    n += 1


optimum = traincosts1.index(min(traincosts1))  
print 'the performance in training set is optimal after ' + str(optimum*100) + ' mini batch gradient descents'
print 'the corresponding correct classification rate on test set is ' + str(testperformance1[optimum]) + '%'

# Identify correct and incorrect cases from test set using the results when training performance is optimal
bestprob = softmax(linComb(testset, bestW, bestb))
bestprediction = bestprob.argmax(axis=0)
results = (bestprediction - testtruth)
successIndices = where(results==0)[0]
successCases = testset.T[random.choice(successIndices, 20)].reshape((4, 5, 784))
correct1, axarr = plt.subplots(4, 5)
for i in range(4):
    for j in range(5):
        fig = axarr[i, j].imshow(successCases[i][j].reshape((28,28)), cmap=cm.gray)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
correct1.savefig('part5 correct.png')
plt.close("all")

failureIndices = nonzero(results)[0]
failureCases = testset.T[random.choice(failureIndices, 10)].reshape((2, 5, 784))
incorrect1, axarr = plt.subplots(2, 5)
for i in range(2):
    for j in range(5):
        fig = axarr[i, j].imshow(failureCases[i][j].reshape((28,28)), cmap=cm.gray)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
incorrect1.savefig('part5 incorrect.png', bbox_inches='tight')
plt.close("all")

# Plotting performances
print 'Plotting the performances on Test and Training Sets...'
plt.figure(1)
plt.plot(range(0, 5001, 100), trainperformance1, 'r')
plt.plot(range(0, 5001, 100), testperformance1, 'b')
plt.ylabel('Correct Classification Rate(%)')
plt.xlabel('Number of gradient descent steps')
plt.yticks(np.arange(30, 105, 5.0))
red_patch = mpatches.Patch(color='red', label='training set')
blue_patch = mpatches.Patch(color='blue', label='test set')
plt.legend(handles=[red_patch, blue_patch], loc='lower right')
plt.title('Correct Digit Classification Rate - 2-layer Neural Network')
plt.savefig('2layer performance.pdf', bbox_inches='tight') 
print 'Results saved as 2layer performance.pdf.' 
plt.close("all")

print 'Plotting the cost functions on Test and Training Sets...'
plt.figure(2)
plt.plot(range(0, 5001, 100), traincosts1, 'r')
plt.plot(range(0, 5001, 100), testcosts1, 'b')
plt.ylabel('negative sum of log probabilities')
plt.xlabel('Number of gradient descent steps')
red_patch = mpatches.Patch(color='red', label='training set')
blue_patch = mpatches.Patch(color='blue', label='test set')
plt.legend(handles=[red_patch, blue_patch])
plt.title('Cost function - 2-layer Neural Network')
plt.savefig('2layer costs.pdf', bbox_inches='tight') 
print 'Results saved as 2layer costs.pdf.'
plt.close("all")


#Part 6
for i in range(10):
    fig = figure(3)
    ax = fig.gca()    
    heatmap = ax.imshow(bestW.T[i][:].reshape((28,28)), cmap = cm.coolwarm)    
    fig.colorbar(heatmap, shrink = 0.5, aspect=5)
    savefig('W output ' + str(i), bbox_inches='tight')
    close(fig)



#Part7
def tanh_layer(y, W, b):    
    '''Return the output of a tanh layer for the input matrix y. y
    is an NxM matrix where N is the number of inputs for a single case, and M
    is the number of cases'''
    return tanh(dot(W.T, y)+b)

def forward(x, W0, b0, W1, b1):
    L0 = tanh_layer(x, W0, b0)
    L1 = tanh_layer(L0, W1, b1)
    output = softmax(L1)
    return L0, L1, output

def deriv_multilayer(W0, b0, W1, b1, x, L0, L1, y, y_):
    '''Return the derivatives of the cost in a multilayer neural network with 
    respect to weight matrices W0, W1 and biases b0, b1. x is the input cases of 
    dimension 784xM, M is the number of cases. L0 and L1 are the outputs in 
    second and third layers of the neural network. y is the matrix of 
    probabilities for each digit and each case. y_ is the one-hot encoded matrix
    .'''
    dCdL1 =  y - y_    
    dCdW1 =  dot(L0, ((1- L1**2)*dCdL1).T )    
    dCdb1 = (dot(ones(L0.shape[1]), ((1- L1**2)*dCdL1).T)).reshape(len(b1),1)
    dCdL0 = dot(W1, ((1- L1**2)*dCdL1))
    dCdW0 = dot(x, ((1- L0**2)*dCdL0).T)
    dCdb0 = dot(ones(x.shape[1]), ((1- L0**2)*dCdL0).T).reshape(len(b0),1)
    return dCdW1, dCdb1, dCdW0, dCdb0
    


#part 8
def appderiv_multilayer(x, W0, b0, W1, b1, y_, d):
    L0, L1, y = forward(x, W0, b0, W1, b1)
    fixpoint = cost(y, y_)
    dCdW1 = zeros(W1.shape)
    dCdb1 = zeros(W1.shape[1])
    dCdW0 = zeros(W0.shape)
    dCdb0 = zeros(W0.shape[1])
    for i in range(W1.shape[1]):
        for j in range(W1.shape[0]):
            W1d = copy(W1).astype(float)
            W1d[j][i] += d
            y = forward(x, W0, b0, W1d, b1)[2]
            dCdW1[j][i] = (cost(y,y_)-fixpoint)/d
    for i in range(W0.shape[1]):
        for j in range(W0.shape[0]):
            W0d = copy(W0).astype(float)
            W0d[j][i] += d
            y = forward(x, W0d, b0, W1, b1)[2]
            dCdW0[j][i] = (cost(y,y_)-fixpoint)/d
    for i in range(W1.shape[1]):
        b1d = copy(b1).astype(float)
        b1d[i] += d
        y = forward(x, W0, b0, W1, b1d)[2]
        dCdb1[i] = (cost(y,y_)-fixpoint)/d
    for i in range(W0.shape[1]):
        b0d = copy(b0).astype(float)
        b0d[i] += d
        y = forward(x, W0, b0d, W1, b1)[2]
        dCdb0[i] = (cost(y,y_)-fixpoint)/d
    return dCdW1, dCdb1.reshape(dCdb1.shape[0],1), dCdW0, dCdb0.reshape(dCdb0.shape[0], 1)

print 'Verifying the gradient calculation using random coods of small dimension'
random.seed(0)
for n in range(3):
    W1 = random.rand(3,2)
    b1 = random.rand(2).reshape(2, 1)
    W0 = random.rand(4,3)
    b0 = random.rand(3).reshape(3, 1)
    x = random.rand(4,3)
    y_ = array([[0, 0, 1], [1, 1, 0]])
    print "the actual gradient calculated"
    L0, L1, y = forward(x, W0, b0, W1, b1)
    dCdW1, dCdb1, dCdW0, dCdb0 = deriv_multilayer(W0, b0, W1, b1, x, L0, L1, y, y_)
    print "dCdW1 = "+ str(dCdW1) 
    print "dCdb1 = " + str(dCdb1)
    print "dCdW0 = "+ str(dCdW0) 
    print "dCdb0 = " + str(dCdb0)
    
    print "the gradient obtained by finite-difference approximation"
    dCdW1, dCdb1, dCdW0, dCdb0 = appderiv_multilayer(x, W0, b0, W1, b1, y_, 0.01)
    print "dCdW1 = "+ str(dCdW1) 
    print "dCdb1 = " + str(dCdb1)
    print "dCdW0 = "+ str(dCdW0) 
    print "dCdb0 = " + str(dCdb0)




#part9
def record2(W0, b0, W1, b1, traincosts, testcosts, trainperformance, testperformance):
    trainprob = forward(trainset, W0, b0, W1, b1)[2]
    trainprediction = trainprob.argmax(axis=0)
    incorrect = count_nonzero(trainprediction - traintruth)
    trainperformance.append((float(trainset.shape[1])-incorrect)*100/(trainset.shape[1]))
    traincosts.append(cost(trainprob,trainmatrix))
    
    testprob = forward(testset, W0, b0, W1, b1)[2]
    testprediction = testprob.argmax(axis=0)
    incorrect = count_nonzero(testprediction - testtruth)
    testperformance.append((float(testset.shape[1])-incorrect)*100/(testset.shape[1]))
    testcosts.append(cost(testprob,testmatrix))


traincosts2 = []
trainperformance2 = []
testperformance2 = []
testcosts2 = []

W0=W0sample
W1=W1sample
b0=b0sample
b1=b1sample
random.seed(0)
n = 0
while n<=10000:
    
    if n%100 == 0:
        print 'gradient descent iteration ' + str(n)
        record2(W0, b0, W1, b1, traincosts2, testcosts2, trainperformance2, testperformance2)
        if traincosts2[-1] <= amin(traincosts2):
            bestW0 = W0
            bestW1 = W1
            bestb0 = b0
            bestb1 = b1
    chosen = random.choice(range(trainset.shape[1]), 50)
    x = (trainset.T[[chosen]]).T
    y_ = (trainmatrix.T[[chosen]]).T 
    L0, L1, y = forward(x, W0, b0, W1, b1)
    dCdW1, dCdb1, dCdW0, dCdb0 = deriv_multilayer(W0, b0,W1, b1, x, L0, L1, y, y_)
    W1 = W1 - alpha*dCdW1
    b1 = b1 - alpha*dCdb1
    W0 = W0 - alpha*dCdW0
    b0 = b0 - alpha*dCdb0
    n += 1


optimum2 = traincosts2.index(min(traincosts2)) 
print 'the cost function in training set is the smallest after ' + str(optimum2*100) + ' mini batch gradient descents'
print 'the corresponding correct classification rate on test set is ' + str(testperformance2[optimum2]) + '%'


# Identify correct and incorrect cases 
bestprob2 = forward(testset, bestW0, bestb0, bestW1, bestb1)[2]
bestprediction2 = bestprob2.argmax(axis=0)
results2 = (bestprediction2 - testtruth)
successIndices2 = where(results2==0)[0]
successCases2 = testset.T[random.choice(successIndices2, 20)].reshape((4, 5, 784))
correct2, axarr = plt.subplots(4, 5)
for i in range(4):
    for j in range(5):
        fig = axarr[i, j].imshow(successCases2[i][j].reshape((28,28)), cmap=cm.gray)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
correct2.savefig('part9 correct.png')

failureIndices2 = nonzero(results2)[0]
failureCases2 = testset.T[random.choice(failureIndices2, 10)].reshape((2, 5, 784))
incorrect2, axarr = plt.subplots(2, 5)
for i in range(2):
    for j in range(5):
        fig = axarr[i, j].imshow(failureCases2[i][j].reshape((28,28)), cmap=cm.gray)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
incorrect2.savefig('part9 incorrect.png')

#Plotting the performances 
print 'Plotting the new performances on Test and Training Sets...'
plt.figure(4)
plt.plot(range(0, 10001, 100), trainperformance2, 'r')
plt.plot(range(0, 10001, 100), testperformance2, 'b')
plt.ylabel('Correct Classification Rate(%)')
plt.xlabel('Number of gradient descent steps')
plt.yticks(np.arange(30, 105, 5.0))
red_patch = mpatches.Patch(color='red', label='training set')
blue_patch = mpatches.Patch(color='blue', label='test set')
plt.legend(handles=[red_patch, blue_patch], loc='lower right')
plt.title('Correct Digit Classification Rate - Multi-layer Neural Network')
plt.savefig('multilayer performance.pdf', bbox_inches='tight') 
print 'Results saved as multilayer performance.pdf.' 
plt.close("all")

print 'Plotting the new cost functions on Test and Training Sets...'
plt.figure(5)
plt.plot(range(0, 10001, 100), traincosts2, 'r')
plt.plot(range(0, 10001, 100), testcosts2, 'b')
plt.ylabel('negative sum of log probabilities')
plt.xlabel('Number of gradient descent steps')
red_patch = mpatches.Patch(color='red', label='training set')
blue_patch = mpatches.Patch(color='blue', label='test set')
plt.legend(handles=[red_patch, blue_patch])
plt.title('Cost function - Multi-layer Neural Network')
plt.savefig('multilayer costs.pdf', bbox_inches='tight') 
print 'Results saved as multilayer costs.pdf.'
plt.close("all")



#Part 10
print 'producing heatmaps of W0 weights connecting inputs to 2 hidden units'

fig = figure(6)
ax = fig.gca()    
heatmap = ax.imshow(bestW0.T[150][:].reshape((28,28)), cmap = cm.coolwarm)    
fig.colorbar(heatmap, shrink = 0.5, aspect=5)
savefig('part 10-1.png', bbox_inches='tight')
close(fig)
print 'the weight connecting the first hidden unit to the output is ', bestW1[150]

fig = figure(7)
ax = fig.gca()    
heatmap = ax.imshow(bestW0.T[270][:].reshape((28,28)), cmap = cm.coolwarm)    
fig.colorbar(heatmap, shrink = 0.5, aspect=5)
savefig('part 10-2.png', bbox_inches='tight')
close(fig)
print 'the weight connecting the second hidden unit to the output is ', bestW1[270]





