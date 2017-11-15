"""p4.py implements text sampling and sentence continuation using minimal 
character-level Vanilla RNN model written by Andrej Karpathy"""

__authors__  = "Wei Zhen Teoh, Ruiwen Li"

# CSC321 Project 4

import numpy as np
import os
import cPickle

# data I/O
data = open('shakespeare_train.txt', 'r').read() # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print 'data has %d characters, %d unique.' % (data_size, vocab_size)
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# hyperparameters
hidden_size = 250 # size of hidden layer of neurons


#Part1

def sample(h, seed_ix, n, temp):
    """ 
    Sample a sequence of integers from the model and record the average hidden 
    state. 
    h is memory state, seed_ix is seed letter for first time step
    , temp is the temperature and n is the character length of text to be 
    generated.
    """
    x = np.zeros((vocab_size, 1))
    x[seed_ix] = 1
    ixes = []
    h_record = np.zeros((hidden_size, 1))
    for t in xrange(n):
        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
        y = np.dot(Why, h) + by
        p = np.exp(y/temp) / np.sum(np.exp(y/temp))
        ix = np.random.choice(range(vocab_size), p=p.ravel())
        x = np.zeros((vocab_size, 1))
        x[ix] = 1
        ixes.append(ix)
        h_record += h
    average_h = h_record/n
    return ixes, average_h

np.random.seed(42)


print 'Generate sentences at various temperatures'




snapshot = cPickle.load(open("char-rnn-snapshot.pkl"))

Wxh, Whh, Why = snapshot['Wxh'], snapshot['Whh'], snapshot['Why']
bh, by = snapshot['bh'], snapshot['by']
h0 = np.random.rand(hidden_size,1)*.1


for t in [1.5, 1.25, 1., .75, .5, .25]:
    print 'temperature = ' + str(t)
    sample_ix = sample(h0, 20, 200, t)[0]
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    print '----\n %s \n----' % (txt, )
    sample_ix = sample(h0, 40, 200, t)[0]
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    print '----\n %s \n----' % (txt, )
    


#Part2

ave_h = sample(h0, 20, 200, 1.)[1]

def complete(hstart, starter, n, temp):
    """
    Complete a starter string with n new characters and return the completed 
    string.
    hstart is the memory state, starter is a starter string, n is the number of 
    characters to be generated subsequently, temp is the temperature for 
    generating subsequent characters 
    """
    add_ix = []
    inputs = [char_to_ix[ch] for ch in starter]
    h = hstart
    for t in xrange(len(inputs)-1):
        x = np.zeros((vocab_size,1)) # encode in 1-of-k representation
        x[inputs[t]] = 1
        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h)+ bh)

    sample_ix = sample(h, char_to_ix[starter[-1]], n, temp)[0] 
    continuation = ''.join(ix_to_char[ix] for ix in sample_ix)
    return starter + continuation
    
np.random.seed(90)

print "At temperature = .75, complete the sentence with given stater strings"

print '------'
starter1 = 'she is married t'
print "Starter string 1:" 
print starter1
print 'Output:'
print complete(ave_h, starter1, 50, .75)

print '------'
starter2 = 'ELIZABETH:\nWe are poor, very po'
print "Starter string 2:" 
print starter2
print 'Output:'
print complete(ave_h, starter2, 50, .75)

print '------'
starter3 = 'Anne is married to me. She is my wi'
print "Starter string 3:" 
print starter3
print 'Output:'
print complete(ave_h, starter3, 50, .75)

print '------'
starter4 = 'England great again'
print "Starter string 4:" 
print starter4
print 'Output:'
print complete(ave_h, starter4, 50, .75)

print '------'
starter5 = 'In those parts beyond the sea,'
print "Starter string 5:" 
print starter5
print 'Output:'
print complete(ave_h, starter5, 50, .75)
print '------'

#Part3

x = np.zeros((vocab_size, 1))
x[char_to_ix[':']] = 1
h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, ave_h) + bh)
y = np.dot(Why, h) + by
p = np.exp(y) / np.sum(np.exp(y))
compare_set = []

print "When the current input is ':'"
for i in range(250):
    if h[i][0]*Why[0][i]>1.5 or h[i][0]*Why[2][i]>1.5:
    #Identify the h and Why elements that contribute most positively to the output for y_0 and y_2
    #0 is the index for newline, 2 for space
        print "---"
        compare_set.append(i)
        #those elements are to be compared when x is other input
        print 'i = ' + str(i)
        print 'h[i] is ' + str(h[i][0])
        print "Wxh[i][9] is " + str(Wxh[i, char_to_ix[':']])
        #9 is the index for character ':'
        print "Why[0][i] is " + str(Why[0][i])
        print "Why[2][i] is " + str(Why[2][i])
        print "h[i]*Why[0][i] is " + str(h[i][0]*Why[0][i])
        print "h[i]*Why[2][i] is " + str(h[i][0]*Why[2][i])
        

x_neutral = np.zeros((vocab_size, 1))
x_neutral += 1./62
h_neutral = np.tanh(np.dot(Wxh, x_neutral) + np.dot(Whh, ave_h) + bh)
y_neutral = np.dot(Why, h_neutral) + by

print "Otherwise given a average across all other characters as input"
for i in compare_set:
        print "---"
        print 'i = ' + str(i)
        print 'h[i] is ' + str(h_neutral[i][0])
        print "Why[0][i] is " + str(Why[0][i])
        print "Why[2][i] is " + str(Why[2][i])
        print "h[i]*Why[0][i] is " + str(h_neutral[i][0]*Why[0][i])
        print "h[i]*Why[2][i] is " + str(h_neutral[i][0]*Why[2][i])


  
  

