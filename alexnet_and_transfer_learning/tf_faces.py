__author__ = "Wei Zhen Teoh, Ruiwen Li"

# The code for setting up AlexNet architecture is contributed by Michael Guerzhoy

from numpy import *
import os
from pylab import *
import numpy as np
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

from scipy.io import loadmat
import tensorflow as tf



# Setting up training, validation and test sets
def rgb2gray(rgb):
    '''Return the grayscale version of the RGB image rgb as a 2D numpy array
    whose range is 0..1
    Arguments:
    rgb -- an RGB image, represented as a numpy array of size n x m x 3. The
    range of the values is 0..255
    '''
    
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray/255.


def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result

"""
testfile = urllib.URLopener() 


training_1 = {}
validation_1 = {}
testing_1 = {}
training_2 = {}
validation_2 = {}
testing_2 = {}
if not os.path.exists("uncropped"):
    os.makedirs("uncropped")
    
for a in act:
    name = a.split()[1].lower()
    i = 0
    x1 = []
    x2 = []
    x3 = []
    x4 = []
    x5 = []
    x6 = []
    for line in open("faces_subset.txt"):
        if a in line:
            bound = line.split()[5].split(',')
            filename = name + '_' + str(i) + '.' + line.split()[4].split('.')[-1]
            #A version without timeout (uncomment in case you need to 
            #unsupress exceptions, which timeout() does)
            #testfile.retrieve(line.split()[4], "uncropped/"+filename)
            #timeout is used to stop downloading images which take too long to download
            
            timeout(testfile.retrieve, 
                    (line.split()[4], "uncropped/" + filename), {}, 100)
            if not os.path.isfile("uncropped/" + filename):
                continue
            print filename
            try:
                f = open("uncropped/" + filename)
                if  line.split()[6] == sha256(f.read()).hexdigest():
                    img = imread("uncropped/" + filename)
                    bound = line.split()[5].split(',')
                    img = img[int(bound[1]):int(bound[3]), int(bound[0]):int(bound[2])]
                    img1 = imresize(img, [227,227])
                    img2 = rgb2gray(img)
                    img2 = imresize(img2, [64,64])
                    if shape(img1) == (227, 227, 3):
                        if i < 70:
                            x1.append(img1)
                            x4.append(img2.flatten())
                        elif i < 95:
                            x2.append(img1)
                            x5.append(img2.flatten())
                        else:
                            x3.append(img1)
                            x6.append(img2.flatten())
                    else:
                        f.close()
                        continue
                    f.close()
                else:
                    f.close()
                    continue
            except:
                f.close()
                continue
            i += 1
    training_2[a] = asarray(x1)
    validation_2[a] = asarray(x2)
    testing_2[a] = asarray(x3)
    training_1[a] = asarray(x4)
    validation_1[a] = asarray(x5)
    testing_1[a] = asarray(x6)
savemat("train_part1", training_1)
savemat("validation_part1", validation_1)
savemat("test_part1", testing_1)
savemat("train_part2", training_2)
savemat("validation_part2", validation_2)
savemat("test_part2", testing_2)

"""

train1 = loadmat("train_part1.mat")
train = loadmat("train_part2.mat")

trainset1 = ((vstack((train1['Angie Harmon'], train1['Gerard Butler'], train1['Daniel Radcliffe'], train1['Lorraine Bracco'], train1['Peri Gilpin'], train1['Michael Vartan'])))/255.).astype(float32)

trainset = ((vstack((train['Angie Harmon'], train['Gerard Butler'], train['Daniel Radcliffe'], train['Lorraine Bracco'], train['Peri Gilpin'], train['Michael Vartan'])))/255.).astype(float32)
                    
harmon = tile(array([1,0,0,0,0,0]), (70,1))
butler = tile(array([0,1,0,0,0,0]), (70,1))
radcliffe = tile(array([0,0,1,0,0,0]), (70,1))
bracco = tile(array([0,0,0,1,0,0]), (70,1))
gilpin = tile(array([0,0,0,0,1,0]), (70,1))
vartan = tile(array([0,0,0,0,0,1]), (70,1))
train1hot = (vstack((harmon, butler, radcliffe, bracco, gilpin, vartan))).astype(float32)

val1 = loadmat("validation_part1.mat")
val = loadmat("validation_part2.mat")

valset1 = ((vstack((val1['Angie Harmon'], val1['Gerard Butler'], val1['Daniel Radcliffe'], val1['Lorraine Bracco'], val1['Peri Gilpin'], val1['Michael Vartan'])))/255.).astype(float32)

valset = ((vstack((val['Angie Harmon'], val['Gerard Butler'], val['Daniel Radcliffe'], val['Lorraine Bracco'], val['Peri Gilpin'], val['Michael Vartan'])))/255.).astype(float32)

vharmon = tile(array([1,0,0,0,0,0]), (25,1))
vbutler = tile(array([0,1,0,0,0,0]), (25,1))
vradcliffe = tile(array([0,0,1,0,0,0]), (25,1))
vbracco = tile(array([0,0,0,1,0,0]), (25,1))
vgilpin = tile(array([0,0,0,0,1,0]), (25,1))
vvartan = tile(array([0,0,0,0,0,1]), (25,1))
val1hot = (vstack((vharmon, vbutler, vradcliffe, vbracco, vgilpin, vvartan))).astype(float32)

tes1 = loadmat("test_part1.mat")
tes = loadmat("test_part2.mat")

testset1 = ((vstack((tes1['Angie Harmon'], tes1['Gerard Butler'], tes1['Daniel Radcliffe'], tes1['Lorraine Bracco'], tes1['Peri Gilpin'], tes1['Michael Vartan'])))/255.).astype(float32)

testset = ((vstack((tes['Angie Harmon'], tes['Gerard Butler'], tes['Daniel Radcliffe'], tes['Lorraine Bracco'], tes['Peri Gilpin'], tes['Michael Vartan'])))/255.).astype(float32)

tharmon = tile(array([1,0,0,0,0,0]), (tes['Angie Harmon'].shape[0],1))
tbutler = tile(array([0,1,0,0,0,0]), (tes['Gerard Butler'].shape[0],1))
tradcliffe = tile(array([0,0,1,0,0,0]), (tes['Daniel Radcliffe'].shape[0],1))
tbracco = tile(array([0,0,0,1,0,0]), (tes['Lorraine Bracco'].shape[0],1))
tgilpin = tile(array([0,0,0,0,1,0]), (tes['Peri Gilpin'].shape[0],1))
tvartan = tile(array([0,0,0,0,0,1]), (tes['Michael Vartan'].shape[0],1))
test1hot = (vstack((tharmon, tbutler, tradcliffe, tbracco, tgilpin, tvartan))).astype(float32)

train_x = zeros((1, 227,227,3)).astype(float32)
train_y = zeros((1, 6))
xdim = train_x.shape[1:]
ydim = train_y.shape[1]

#Part1

tf.set_random_seed(24)
random.seed(24)

#Setting up a fully connected Neural Network with 300 hidden units
print 'setting up a 3-layer fully connected neural network with 300 hidden units'

def FCN1(x, W0, W1, b0, b1):
    layer1 = tf.nn.tanh(tf.matmul(x, W0)+b0)
    layer2 = tf.nn.tanh(tf.matmul(layer1, W1)+b1)
    y = tf.nn.softmax(layer2)
    return y

def evaluate(y, y_):
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy
    
def NLLF(lam, W0, W1, y, y_):
    decay_penalty =lam*tf.reduce_sum(tf.square(W0))+lam*tf.reduce_sum(tf.square(W1))
    NLL = -tf.reduce_sum(y_*tf.log(y)+decay_penalty)
    return NLL
    
lam = 0.0002
nhid1 = 300

W0 = tf.Variable(tf.random_normal([4096, nhid1], stddev=0.01))
b0 = tf.Variable(tf.random_normal([nhid1], stddev=0.01))

W1 = tf.Variable(tf.random_normal([nhid1, 6], stddev=0.01))
b1 = tf.Variable(tf.random_normal([6], stddev=0.01))

x = tf.placeholder(tf.float32, [None, 4096])
y_ = tf.placeholder(tf.float32, [None, 6])    
y = FCN1(x, W0, W1, b0, b1)
accuracy = evaluate(y, y_)
NLL = NLLF(lam, W0, W1, y, y_)
train_step = tf.train.GradientDescentOptimizer(0.005).minimize(NLL)

#Performance
def get_train_batch(n, set, onehot):
    chosen = random.choice(range(set.shape[0]), n)
    batch_xs = set[[chosen]]
    batch_y_s = onehot[[chosen]]
    return batch_xs, batch_y_s


trainRate1 = []
valRate1 = []
testRate1 = []

init_vars1 = tf.initialize_variables([W0, b0, W1, b1])

vbest300 = []
rbest300 = 0
with tf.Session() as sess:
    sess.run(init_vars1)

    for i in range(2000):
        batch_xs, batch_y_s = get_train_batch(20, trainset1, train1hot)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_y_s})
        
        if (i+1) % 20 == 0:
            print "iteration=",i+1
            trainRate1.append(sess.run(accuracy, feed_dict={x: trainset1, y_: train1hot}))
            valRate1.append(sess.run(accuracy, feed_dict={x: valset1, y_: val1hot}))
            testRate1.append(sess.run(accuracy, feed_dict={x: testset1, y_: test1hot}))
            
            if argmax(valRate1) == len(valRate1)-1:
                rbest300 = sess.run(accuracy, feed_dict={x: testset1, y_: test1hot})
                vbest300 = [sess.run(W0), sess.run(W1), sess.run(b0), sess.run(b1)]



#Plotting the performances 
print 'Plotting the performances on Test and Training Sets...'
plt.figure()
plt.plot(range(20, 2001, 20), trainRate1, 'r')
plt.plot(range(20, 2001, 20), testRate1, 'b')
plt.plot(range(20, 2001, 20), valRate1, 'g')

plt.ylabel('Correct Classification Rate')
plt.xlabel('Number of gradient descent steps')
plt.yticks(np.arange(0, 1, 0.05))
red_patch = mpatches.Patch(color='red', label='training set')
blue_patch = mpatches.Patch(color='blue', label='test set')
green_patch = mpatches.Patch(color='green', label='validation set')
plt.legend(handles=[red_patch, blue_patch, green_patch])
plt.title('Learning Curve')
plt.savefig('lc300.pdf', bbox_inches='tight')
plt.close("all")

print 'the performance first peaks on validation set after ' + str((argmax(valRate1)+1)*20) + ' gradient descents' 
print 'the successful classification rate on validation set is ' + str(max(valRate1))
print 'the successful classification rate on test set is ' + str(testRate1[argmax(valRate1)])


#Setting up a fully connected neural network with 800 hidden units
print 'setting up a 3-layer fully connected neural network with 800 hidden units'
lam = 0.0002
nhid1 = 800

W0 = tf.Variable(tf.random_normal([4096, nhid1], stddev=0.01))
b0 = tf.Variable(tf.random_normal([nhid1], stddev=0.01))

W1 = tf.Variable(tf.random_normal([nhid1, 6], stddev=0.01))
b1 = tf.Variable(tf.random_normal([6], stddev=0.01))

x = tf.placeholder(tf.float32, [None, 4096])
y_ = tf.placeholder(tf.float32, [None, 6])    
y = FCN1(x, W0, W1, b0, b1)
accuracy = evaluate(y, y_)
NLL = NLLF(lam, W0, W1, y, y_)
train_step = tf.train.GradientDescentOptimizer(0.005).minimize(NLL)


trainRate2 = []
valRate2 = []
testRate2 = []

init_vars1 = tf.initialize_variables([W0, b0, W1, b1])

vbest800 = []
rbest800 = 0
with tf.Session() as sess:
    sess.run(init_vars1)

    for i in range(2000):
        batch_xs, batch_y_s = get_train_batch(20, trainset1, train1hot)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_y_s})
        
        if (i+1) % 20 == 0:
            print "iteration=",i+1
    
            trainRate2.append(sess.run(accuracy, feed_dict={x: trainset1, y_: train1hot}))
            valRate2.append(sess.run(accuracy, feed_dict={x: valset1, y_: val1hot}))
            testRate2.append(sess.run(accuracy, feed_dict={x: testset1, y_: test1hot}))
    
            if argmax(valRate2) == len(valRate2)-1:
                rbest800 = sess.run(accuracy, feed_dict={x: testset1, y_: test1hot})
                vbest800 = [sess.run(W0), sess.run(W1), sess.run(b0), sess.run(b1)]


#Plotting the performances 
print 'Plotting the new performances on Test and Training Sets...'
plt.figure()
plt.plot(range(20, 2001, 20), trainRate1, 'r')
plt.plot(range(20, 2001, 20), testRate1, 'b')
plt.plot(range(20, 2001, 20), valRate1, 'g')

plt.ylabel('Correct Classification Rate')
plt.xlabel('Number of gradient descent steps')
plt.yticks(np.arange(0, 1, 0.05))
red_patch = mpatches.Patch(color='red', label='training set')
blue_patch = mpatches.Patch(color='blue', label='test set')
green_patch = mpatches.Patch(color='green', label='validation set')
plt.legend(handles=[red_patch, blue_patch, green_patch])
plt.title('Learning Curve')
plt.savefig('lc800.pdf', bbox_inches='tight')
plt.close("all")

print 'the performance first peaks on validation set after ' + str((argmax(valRate2)+1)*20) + ' gradient descents' 
print 'the successful classification rate on validation set is ' + str(max(valRate2))
print 'the successful classification rate on test set is ' + str(testRate2[argmax(valRate2)])





#part3
print 'generating heatmaps for trained weights'


fig = figure()
ax = fig.gca()    
heatmap = ax.imshow(vbest300[0].T[54][:].reshape((64,64)), cmap = cm.coolwarm)    
fig.colorbar(heatmap, shrink = 0.5, aspect=5)
savefig('300_1.png', bbox_inches='tight')
close(fig)
print 'weights for second layer connected to trained weight 1 is ' + str(vbest300[1][54])

fig = figure()
ax = fig.gca()    
heatmap = ax.imshow(vbest300[0].T[40][:].reshape((64,64)), cmap = cm.coolwarm)    
fig.colorbar(heatmap, shrink = 0.5, aspect=5)
savefig('300_2.png', bbox_inches='tight')
close(fig)
print 'weights for second layer connected to trained weight 2 is ' + str(vbest300[1][40])
 
fig = figure()
ax = fig.gca()    
heatmap = ax.imshow(vbest800[0].T[450][:].reshape((64,64)), cmap = cm.coolwarm)    
fig.colorbar(heatmap, shrink = 0.5, aspect=5)
savefig('800_1.png', bbox_inches='tight')
close(fig)
print 'weights for second layer connected to trained weight 3 is ' + str(vbest800[1][450])

fig = figure()
ax = fig.gca()    
heatmap = ax.imshow(vbest800[0].T[83][:].reshape((64,64)), cmap = cm.coolwarm)    
fig.colorbar(heatmap, shrink = 0.5, aspect=5)
savefig('800_2.png', bbox_inches='tight')
close(fig)
print 'weights for second layer connected to trained weight 4 is ' + str(vbest800[1][83])




#Part2
# Setting up AlexNet to process images
tf.set_random_seed(42)
random.seed(42)
net_data = load("bvlc_alexnet.npy").item()

train_x = zeros((1, 227,227,3)).astype(float32)
train_y = zeros((1, 6))
xdim = train_x.shape[1:]
ydim = train_y.shape[1]


x_dummy = (random.random((1,)+ xdim)/255.).astype(float32)

def preprocess(face):
    i = x_dummy.copy()
    i[0,:,:,:] = face
    i = i-mean(i)
    return i

def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    
    
    if group==1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(3, group, input)
        kernel_groups = tf.split(3, group, kernel)
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(3, output_groups)
    return  tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())


print 'setting up AlexNet for image transformation'
#reading images
x_in = tf.placeholder(tf.float32, [1, 227, 227, 3])
x = tf.Variable(x_in, trainable=False)

#conv1
#conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
conv1W = tf.Variable(net_data["conv1"][0])
conv1b = tf.Variable(net_data["conv1"][1])
conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
conv1 = tf.nn.relu(conv1_in)

#lrn1
#lrn(2, 2e-05, 0.75, name='norm1')
radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn1 = tf.nn.local_response_normalization(conv1,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

#maxpool1
#max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)


#conv2
#conv(5, 5, 256, 1, 1, group=2, name='conv2')
k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
conv2W = tf.Variable(net_data["conv2"][0])
conv2b = tf.Variable(net_data["conv2"][1])
conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv2 = tf.nn.relu(conv2_in)


#lrn2
#lrn(2, 2e-05, 0.75, name='norm2')
radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn2 = tf.nn.local_response_normalization(conv2,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

#maxpool2
#max_pool(3, 3, 2, 2, padding='VALID', name='pool2')                                                  
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

#conv3
#conv(3, 3, 384, 1, 1, name='conv3')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
conv3W = tf.Variable(net_data["conv3"][0])
conv3b = tf.Variable(net_data["conv3"][1])
conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv3 = tf.nn.relu(conv3_in)

#conv4
#conv(3, 3, 384, 1, 1, group=2, name='conv4')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
conv4W = tf.Variable(net_data["conv4"][0])
conv4b = tf.Variable(net_data["conv4"][1])
conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv4 = tf.nn.relu(conv4_in)

init = tf.initialize_all_variables()

#input original images and collect AlexNet conv4 outputs as new training, test and validation sets
def transform(set):
    newset = zeros((0, 64896))
    for n in range(set.shape[0]):
        preprocessed = preprocess(set[n])
        with tf.Session() as sess1: 
            sess1.run(init, feed_dict={x_in: preprocessed})
            newset = vstack((newset, sess1.run(conv4, feed_dict={x_in: preprocessed}).copy().reshape((1,64896))))
    return newset

newval = transform(valset)
newtrain = transform(trainset)
newtest = transform(testset)


# fully connected Neural Network
print 'setting up a fully connected neural network to receive outputs from AlexNet as new inputs...' 

def FCN(u, W0, W1, b0, b1):
    layer1 = tf.nn.tanh(tf.matmul(u, W0)+b0)
    layer2 = tf.nn.tanh(tf.matmul(layer1, W1)+b1)
    y = tf.nn.softmax(layer2)
    return y
    
lam = 0.00002
nhid = 300
W0 = tf.Variable(tf.random_normal([64896, nhid], stddev=0.08))
b0 = tf.Variable(tf.random_normal([nhid], stddev=0.08))

W1 = tf.Variable(tf.random_normal([nhid, 6], stddev=0.08))
b1 = tf.Variable(tf.random_normal([6], stddev=0.08))

new_x = tf.placeholder(tf.float32, [None, 64896])
y_ = tf.placeholder(tf.float32, [None, 6])    
y = FCN(new_x, W0, W1, b0, b1)
rate = evaluate(y, y_)
NLL = NLLF(lam, W0, W1, y, y_)
train_step = tf.train.GradientDescentOptimizer(0.005).minimize(NLL)


#Performance
print 'training the fully conncted network and recording the performance...'
init_new_vars = tf.initialize_variables([W0, b0, W1, b1])

trainRate = []
valRate = []
testRate = []
vbest = []
with tf.Session() as sess2:
    sess2.run(init_new_vars)

    for i in range(2000):
        batch_xs, batch_y_s = get_train_batch(30, newtrain, train1hot)
        sess2.run(train_step, feed_dict={new_x: batch_xs, y_: batch_y_s})

  
        if (i+1) % 20 == 0:
            print "iteration=",i+1
        
            trainRate.append(sess2.run(rate, feed_dict={new_x: newtrain, y_:                train1hot}))
            valRate.append(sess2.run(rate, feed_dict={new_x: newval, y_: val1hot}))
            testRate.append(sess2.run(rate, feed_dict={new_x: newtest, y_: test1hot}))
    
            if argmax(valRate) == len(valRate)-1:
                vbest = [sess2.run(W0), sess2.run(W1), sess2.run(b0), sess2.run(b1)]
    
#Plotting the performances 
print 'Plotting the new performances on Test and Training Sets...'
plt.figure()
plt.plot(range(20, 2001, 20), trainRate, 'r')
plt.plot(range(20, 2001, 20), testRate, 'b')
plt.plot(range(20, 2001, 20), valRate, 'g')

plt.ylabel('Correct Classification Rate')
plt.xlabel('Number of gradient descent steps')
plt.yticks(np.arange(0, 1, 0.05))
red_patch = mpatches.Patch(color='red', label='training set')
blue_patch = mpatches.Patch(color='blue', label='test set')
green_patch = mpatches.Patch(color='green', label='validation set')
plt.legend(handles=[red_patch, blue_patch, green_patch])
plt.title('Learning Curve')
plt.savefig('lc_part2.png', bbox_inches='tight')
plt.close("all")

print 'the performance first peaks on validation set after ' + str((argmax(valRate)+1)*20) + ' gradient descents' 
print 'the successful classification rate on validation set is ' + str(max(valRate))
print 'the successful classification rate on test set is ' + str(testRate[argmax(valRate)])

snapshot = {}
snapshot["part2 weights"] = vbest
cPickle.dump(snapshot,  open("record"+".pkl", "w"))





#Part4

print 'setting up an integrated system with AlexNet + fully connected neural netwrok' 
#we reuse the best weights obtained from part 2
lam = 0.00002
nhid = 300
W0 = tf.Variable(vbest[0])
b0 = tf.Variable(vbest[2])
W1 = tf.Variable(vbest[1])
b1 = tf.Variable(vbest[3])

#Modification for AlexNet
reshaped = tf.reshape(conv4, [1, 64896])
probs = FCN(reshaped, W0, W1, b0, b1)
#probs is a tensor object that will record the probabilities 
#for each class of an input image

print 'An image of Angie Harmon from the test set is loaded into the integrated system and the following is obtained:'
#Case example - print probabilities for each label for an image input
candidates = ['Angie Harmon', 'Gerard Butler', 'Daniel Radcliffe', 'Lorraine Bracco', 'Peri Gilpin', 'Michael Vartan']

init = tf.initialize_all_variables()

with tf.Session() as sess4:
    sess4.run(init, feed_dict={x_in: preprocess(testset[10])})
    guess = sess4.run(probs, feed_dict={x_in: preprocess(testset[10])})[0]
    order = sorted(range(len(guess)), key = lambda x: guess[x], reverse = True)
    orderedguess = sorted(guess, reverse=True)
    for i in range(6):
        print candidates[order[i]], orderedguess[i]



#part5
print 'the gradient of the probability output for the class Angie Harmon with respect to the image just now is obtained and saved.' 
onehot = tf.Variable(test1hot[10].reshape(6,1))
output = tf.matmul(probs, onehot)
#output only records the probability of the correct class 

grad = tf.gradients(output, x)

init = tf.initialize_all_variables()

with tf.Session() as sess5:
    sess5.run(init, feed_dict={x_in: preprocess(testset[10])})
    gradient =  sess5.run(grad, feed_dict={x_in: preprocess(testset[10])})[0][0]

gradient[gradient < 0.] = 0.
gradient = 30*gradient/norm(gradient)
#scaling the gradient to improve image appearance

imsave('face.png', testset[10])
imsave('gradient.png', gradient)
    

    
    
    
    
    

        
