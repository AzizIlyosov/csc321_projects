"""faces.py implements K-Nearest Neighbours algorithm for face recognition tasks
including name and gender classifications."""

__author__  = "Wei Zhen Teoh"

from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import matplotlib.patches as mpatches
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
from scipy.misc import imsave
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib
from collections import Counter

gray()

def rgb2gray(rgb):    
    """return the grayscaled image of rgb""" 
    if len(rgb.shape) == 2:
        return rgb    
    else:
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray/255

def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/
    '''    
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

testfile = urllib.URLopener()            

#subset_actors.txt and subset_actresses.txt are first downloaded to the 
#working directory
testfile.retrieve\
('http://www.cs.toronto.edu/~guerzhoy/321/proj1/subset_actors.txt',\
 'subset_actors.txt')
testfile.retrieve\
('http://www.cs.toronto.edu/~guerzhoy/321/proj1/subset_actresses.txt',\
 'subset_actresses.txt')

#'uncropped' and 'processed' folders are created 
#to store the downloaded images and the processed images
os.mkdir('uncropped')
os.mkdir('processed')

print 'Downloading and processing the images...'

act = ['Gerard Butler', 'Daniel Radcliffe', 'Michael Vartan']
acts = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon']

#actors' images are downloaded to uncropped folder
#faces are cropped out from the images, grayscaled, resized and saved in 
#processed folder 
for a in act:
    name = a.split()[-1].lower()
    i = 1
    for line in open("subset_actors.txt"):
        if i <= 120:
            if a in line: 
                ls = line.split()
                #bbox records the measurements of bounding boxes for faces
                bbox = ls[-2].split(',') 
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), \
                                     int(bbox[3])
                filename = name+str(i)+'.png'
                #timeout is used to stop downloading images which take too 
                #long to download
                timeout(testfile.retrieve, (ls[4], "uncropped/"+filename), \
                        {}, 30)
                if not os.path.isfile("uncropped/"+filename):
                    continue
                
                try:
                    #the image is tested to ensure that the download link is 
                    #not broken, the downloaded file is replaced by a new image 
                    #otherwise                    
                    testpic = imread("uncropped/"+filename)
                    if testpic.shape == (1L,1L):
                        continue
                    print filename
                    #face is cropped out from the image based on bounding box 
                    #measurements
                    cropped = testpic[y1:y2, x1:x2]
                    #face (image) is grayscaled and resized
                    gscaled = rgb2gray(cropped)
                    resized = imresize(gscaled, (32,32))
                    imsave('processed/'+filename, resized)
                    i = i + 1
                     
                except:
                    continue

#actresses' images are downloaded in uncropped folder
#faces are cropped out from the images, grayscaled, resized and saved in 
#processed folder
for a in acts:
    name = a.split()[-1].lower()
    i = 1
    for line in open("subset_actresses.txt"):
        if i <= 120:
            if a in line: 
                ls = line.split()
                bbox = ls[-2].split(',') 
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]),\
                                     int(bbox[3])
                filename = name+str(i)+'.png'
                timeout(testfile.retrieve, (ls[4], "uncropped/"+filename), \
                        {}, 30)
                if not os.path.isfile("uncropped/"+filename):
                    continue
                
                try:
                    testpic = imread("uncropped/"+filename)
                    if testpic.shape == (1L,1L):
                        continue
                    print filename
                    cropped = testpic[y1:y2, x1:x2]
                    gscaled = rgb2gray(cropped)
                    resized = imresize(gscaled, (32,32))
                    imsave('processed/'+filename, resized)
                    i = i + 1
                     
                except:
                    continue





#Part 2 Setting up training set, validation set and test set 

class face(object):
    """A face records information associated with a cropped out face image 
    in processed folder. The attributes record the image itself, the vector 
    obtained by flattening the image 2D-array, the person's name and gender.
    
    The guess attribute records the KNN prediction of the person's name (or the 
    person's gender) using only the vector as input.
    
    The fiveNN attribute records a list of five nearest neighbour faces in 
    the training set based on the L2-distance between the vectors.
    """  
    
    def __init__(self, image, name, gender):
        self.image = image
        self.vector = image.flatten()  
        self.name = name
        self.gender = gender
        self.guess = None
        self.fiveNN = None

    def updateguess(self, guess):
        self.guess = guess
        
    def updatefiveNN(self, fiveNN):
        self.fiveNN = fiveNN
    
#Group the faces according to the person's names
butler = []
radcliffe = []
vartan = []
bracco = []
gilpin = []
harmon = []

#A face instance is created for each face image in the processed folder 
#the name and gender are also recorded for labelling (training set) and KNN 
#prediction verification to evaluate performance (validation and test sets)
os.chdir('processed')
for a in act:
    filename = a.split()[-1].lower() 
    i = 1
    while i <= 120:
        image = imread(filename+ str(i)+ '.png')
        faceInstance = face(image, a, 'male')
        if a == 'Gerard Butler':
            butler.append(faceInstance)
        elif a == 'Daniel Radcliffe':
            radcliffe.append(faceInstance)
        elif a == 'Michael Vartan':
            vartan.append(faceInstance)
        i += 1
        
for a in acts:
    filename = a.split()[-1].lower() 
    i = 1
    while i <= 120:
        image = imread(filename+ str(i)+ '.png')
        faceInstance = face(image, a, 'female')
        if a == 'Lorraine Bracco':
            bracco.append(faceInstance)
        elif a == 'Peri Gilpin':
            gilpin.append(faceInstance)
        elif a == 'Angie Harmon':
            harmon.append(faceInstance)
        i += 1

#split the faces into 3 non-overlapping sets: 
#training, validation and test set
print 'Setting up the training, validation and test set...'
trainset = butler[10:110] + radcliffe[10:110] + vartan[10:110] \
           + bracco[10:110] + gilpin [10:110] + harmon[10:110] 
valset = butler[110:120] + radcliffe[110:120] + vartan[110:120] \
         + bracco[110:120] + gilpin [110:120] + harmon[110:120]
testset = butler[0:10] + radcliffe[0:10] + vartan[0:10] \
            + bracco[0:10] + gilpin [0:10] + harmon[0:10]





# Part 3 KNN Face Recognition and choosing the best K 
print \
'Recognizing faces from the processed images...'

def distance(face1, face2):
    """Return the L2-distance between the vectors of face1 and face2
    """
    return l2norm(face1.vector - face2.vector)

def selectionSort(neighbours, face):
    """Order the neighbour faces in the neighbours list according to their 
    distance from the face (from the nearest neighbour to the furthest 
    neighbour)."""
    for fillslot in range(len(neighbours)-1,0,-1):
        furthest = 0
        for index in range(1,fillslot+1):
            if distance(neighbours[index], face)> \
               distance(neighbours[furthest], face):
                furthest = index

        temp = neighbours[fillslot]
        neighbours[fillslot] = neighbours[furthest]
        neighbours[furthest] = temp

def updateNeighbours(neighbours, newNeighbour, face):
    """Update the neighbours list depending on the distance of the newNeighbour 
    from the face. newNeighbour is added into the order if it is nearer to the 
    face than some of the current neighbours. In that case the furthest current 
    neighbour is discarded."""
    j = len(neighbours)
    while j != 0 and distance(newNeighbour, face) <\
          distance(neighbours[j-1], face):
        j = j - 1
    neighbours.insert(j, newNeighbour)
    del neighbours[-1]

def kNN(face, searchset, num):
    """Return num nearest neighbours in the searchset to the face in ascending
    order (based on the distance)"""
    neighbours = searchset[0:num]
    selectionSort(neighbours, face)
    i = num
    while i < len(searchset):
        updateNeighbours(neighbours, searchset[i], face)
        i += 1
    return neighbours
    
def namevote(neighbours):
    """Return the name with highest count among the neighbours. If there is a 
    tie an arbitrary choice will be made among those names with the highest 
    count."""
    neighbourNames = []
    for n in neighbours:
        neighbourNames.append(n.name)
    return Counter(neighbourNames).most_common(1)[0][0]

print 'Measuring the performance on the training set. This may take a while...'
trainPerformance = []
#the performance is measured for odd K between 0 and 40
for k in range(1,40,2):
    correct = 0
    for trainface in trainset:
        neighbours = kNN(trainface, trainset, k)
        #the trainface is labelled with KNN guess
        trainface.updateguess(namevote(neighbours))
        #the correctness of the guess is verified
        if trainface.guess == trainface.name:
            correct += 1
    #success rate on each K is calculated and recorded
    trainPerformance.append(100 * float(correct)/len(trainset))

print 'Measuring the performance on the validation set...'
valPerformance = []
for k in range(1,40,2):
    correct = 0
    for valface in valset:
        neighbours = kNN(valface, trainset, k)
        valface.updateguess(namevote(neighbours))
        if valface.guess == valface.name:
            correct += 1
    valPerformance.append(100 * float(correct)/len(valset))
#bestk records the best K based on the performance on validation set. If there 
#are more than one of such K, the smallest one is chosen
bestk = valPerformance.index(max(valPerformance)) * 2 + 1 
print 'The performance on the validation set is optimized with K = ' \
       + str(bestk) + '.'

print 'Measuring the performance on the test set...'
failureCount = 0
failureExp = []
failedNames = []
failedGuesses = []
testPerformance = []
for k in range(1,40,2):
    correct = 0    
    for testface in testset:
        neighbours = kNN(testface, trainset, k)            
        testface.updateguess(namevote(neighbours))
        if testface.guess == testface.name:
            correct += 1
        if k == bestk:
            #failed examples of KNN prediction using best K are recorded in 
            #the failureExp list, with the guesses recorded in failedGuesses
            if testface.guess != testface.name:
                failureExp.append(testface)
                failedGuesses.append(testface.guess)
                
        #five nearest neighbours of the test set subject are recorded
        if k == 5:
            testface.updatefiveNN(neighbours)    
    performance = 100 * float(correct)/len(testset)
    if k == bestk:
        print 'The success rate on the test set using the best K is ' +\
              str(performance) + '%.'        
    testPerformance.append(performance)

#a failure folder is created to store face images of five failed examples of
#KNN prediction
os.chdir('..')
os.mkdir('failure')
os.chdir('failure')

#five failed samples are taken from the list of failutreExp
random.seed(24)
failedSample = random.sample(failureExp, 5)
random.seed(24)
failedSampleGuesses = random.sample(failedGuesses, 5)

i=1
for fface in failedSample:
    imsave('exp' + str(i) + '.png', fface.image)
    j = 1
    #the face images of five nearest neighbours to each failed example 
    #are stored in the failure folder too
    for neighbourface in fface.fiveNN:
        imsave('exp' + str(i) + 'NN' + str(j) +'.png', neighbourface.image)
        j += 1 
    i += 1

#we can access the failed examples' actual identities running the following
#failedSample[0].name
#failedSample[1].name
#failedSample[2].name
#failedSample[3].name
#failedSample[4].name

#and the corresponding failed guesses are stored in order in the list 
#failedSampleGuesses 

# Part 4 Plotting the performances (KNN Face Recognition)
os.chdir('..')
print 'Plotting the performances on Validation, Test and Training Set...'
plt.figure(1)
plt.plot(range(1, 40, 2), valPerformance, 'r')
plt.plot(range(1, 40, 2), testPerformance, 'g')
plt.plot(range(1, 40, 2), trainPerformance, 'b')
plt.ylabel('Success Rate(%)')
plt.xlabel('K Chosen')
plt.yticks(np.arange(30, 105, 5.0))
green_patch = mpatches.Patch(color='green', label='test set')
blue_patch = mpatches.Patch(color='blue', label='training set')
red_patch = mpatches.Patch(color='red', label='validation set')
plt.legend(handles=[red_patch, blue_patch, green_patch])
plt.title('KNN Name Classification Performance on 3 Distinct Sets')
plt.savefig('namePerformance.pdf', bbox_inches='tight') 
print 'Results saved as namePerformance.pdf.' 





# Part 5 Classifying gender by faces and evaluate the performances
print 'Classifying gender by faces from the processed images...'

def gendervote(neighbours):
    """Return the gender with the highest count among the neighbours. If there 
    is a tie an arbitrary choice will be made among those names with the highest 
    count."""
    neighbourGenders = []
    for n in neighbours:
        neighbourGenders.append(n.gender)
    return Counter(neighbourGenders).most_common(1)[0][0]

print 'Measuring the performance on the validation set...'
valPerformance2 = []
for k in range(1,40,2):
    correct = 0
    for valface in valset:
        neighbours = kNN(valface, trainset, k)
        #the validation set subject is labelled with KNN gender prediction  
        valface.updateguess(gendervote(neighbours))
        if valface.guess == valface.gender:
            correct += 1
    valPerformance2.append(100 * float(correct)/len(valset))
#bestk2 records the best K based on the performance on validation set. If there 
#are more than one of such K, the smallest one is chosen 
bestk2 = valPerformance2.index(max(valPerformance2)) * 2 + 1 

print 'The performance on the validation set is optimized with K = ' +\
       str(bestk2) + '.'

print 'Plotting the performance on Validation Set...'
plt.figure(2)
plt.plot(range(1, 40, 2), valPerformance2, 'r')
plt.ylabel('Success Rate(%)')
plt.xlabel('K Chosen')
plt.yticks(np.arange(60, 105, 5.0))
plt.title('KNN Gender Classification Performance on Validation Set')
plt.savefig('genderPerformance.pdf', bbox_inches='tight') 
print 'Results saved as genderPerformance.pdf.' 

print 'Measuring the performance on the test set...'
correct = 0
for testface in testset:
    neighbours = kNN(testface, trainset, bestk2)            
    testface.updateguess(gendervote(neighbours))
    if testface.guess == testface.gender:
            correct += 1
performance = 100 * float(correct)/len(testset)
print 'The success rate on the test set using the best K is ' +\
       str(performance) + '%.'





# Part 6 Testing KNN gender classification with new faces   

print 'Downloading and processing the images of other actors and actressess...'

with open('subset_actors.txt') as m:
    mlines = m.read().splitlines()

#60 actors' images are downloaded to uncropped folder 
#and have faces extracted to processed folder
random.seed(42)
maleCount = 1
while maleCount <= 60:
    line = random.choice(mlines)
    ls = line.split()
    if not ls[0] + ' ' + ls[1] in act:
        bbox = ls[-2].split(',') 
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        filename = 'male' + str(maleCount)+ '.png'
        timeout(testfile.retrieve, (ls[4], "uncropped/"+filename), {}, 30)
        if not os.path.isfile("uncropped/"+filename):
            continue
        
        try:
            testpic = imread("uncropped/"+filename)
            if testpic.shape == (1L,1L):
                continue
            cropped = testpic[y1:y2, x1:x2]
            gscaled = rgb2gray(cropped)
            resized = imresize(gscaled, (32,32))
            imsave('processed/'+filename, resized)
            print filename
            maleCount += 1
            
        except:
            continue

with open('subset_actresses.txt') as f:
    flines = f.read().splitlines()

#60 actors' images are downloaded to uncropped folder 
#and have faces extracted to processed folder
femaleCount = 1
while femaleCount <= 60:
    line = random.choice(flines)
    ls = line.split()
    if not ls[0] + ' ' + ls[1] in acts:
        bbox = ls[-2].split(',') 
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        filename = 'female' + str(femaleCount)+ '.png'
        timeout(testfile.retrieve, (ls[4], "uncropped/"+filename), {}, 30)
        if not os.path.isfile("uncropped/"+filename):
            continue
        
        try:
            testpic = imread("uncropped/"+filename)
            if testpic.shape == (1L,1L):
                continue
            cropped = testpic[y1:y2, x1:x2]
            gscaled = rgb2gray(cropped)
            resized = imresize(gscaled, (32,32))
            imsave('processed/'+filename, resized)
            print filename
            femaleCount += 1
            
        except:
            continue

print 'Classifying gender of the new set of actors and actresses....'

#the faces are grouped by gender
male = []
female = []

os.chdir('processed')
for i in range(1,61):
    image = imread('male'+ str(i)+ '.png')
    faceInstance = face(image, 'unknown', 'male')
    male.append(faceInstance)

for i in range(1,61):
    image = imread('female'+ str(i)+ '.png')
    faceInstance = face(image, 'unknown', 'female')
    female.append(faceInstance)

#validation and test set each contains 30 male faces and 30 female faces
print 'Setting up the new validation and test set...' 
valset2 = male[0:30] + female[0:30]
testset2 = male[30:] + female[30:]

print 'Measuring the performance on the new validation set...'
valPerformance3 = []
for k in range(1,40,2):
    correct = 0
    for valface in valset2:
        neighbours = kNN(valface, trainset, k)
        valface.updateguess(gendervote(neighbours))
        if valface.guess == valface.gender:
            correct += 1
    valPerformance3.append(100 * float(correct)/len(valset2))
#bestk3 records the best K based on the performance on validation set. If there 
#are more than one of such K, the smallest one is chosen
bestk3 = valPerformance3.index(max(valPerformance3)) * 2 + 1 

print 'The performance on the new validation set is optimized with K = ' +\
      str(bestk3) + '.'

os.chdir('..')
print 'Plotting the performance on the new validation set...'
plt.figure(3)
plt.plot(range(1, 40, 2), valPerformance3, 'r')
plt.ylabel('Success Rate(%)')
plt.xlabel('K Chosen')
plt.yticks(np.arange(30, 80, 5.0))
plt.title('KNN Gender Classification Performance on new Validation Set')
plt.savefig('newGenderPerformance.pdf', bbox_inches='tight') 
print 'Results saved as newGenderPerformance.pdf.' 

print 'Measuring the performance on the test set...'
correct = 0
for testface in testset2:
    neighbours = kNN(testface, trainset, bestk3)            
    testface.updateguess(gendervote(neighbours))
    if testface.guess == testface.gender:
            correct += 1
performance = 100 * float(correct)/len(testset2)
print 'The success rate on the new test set using the best K is ' +\
      str(performance) + '%.'


 




    
    




    

        
                    
        

            
                    
