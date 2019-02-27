import sys
import pandas as pd
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement

def getAllCombinations(eleList):
    finalList = []
    for i in range(0,len(eleList)):
        print("I is equal to :" + str(i+1))
        comb = combinations(eleList,i+1)
        combi = []
        for j in comb:
            finalList.append(list(j))
    return finalList
        

def gradientDescentCalculation(X,y,theta):
    hypothesis = np.dot(X,theta)
    grad = np.dot(np.transpose(X),np.subtract(hypothesis,y))
    theta = np.subtract(theta, (lr/N)*grad)
    return theta

def normalEquation(X,y):
    inverted = np.linalg.pinv(np.dot(np.transpose(X),X))
    theta = np.dot(np.dot(inverted,np.transpose(X)),y)
    return theta

def computeCost(X,y,theta):
    hypothesis = np.dot(X,theta)
    err = np.subtract(hypothesis,y)
    errSqrd = np.dot(np.transpose(err),err)
    cost = np.multiply(0.5/N,errSqrd)
    return cost

def normalise(data):
    mean = np.mean(data,axis=0)
    std = np.std(data,axis=0)
    for i in range(0,N):
        data[i] = (data[i] - mean) / std
    return data


def addPolynomialFeatures(data,*args):
    i = 0
    for feature,power in args:
        data = np.concatenate((data,data[:,[feature]]**power), axis=1)
        i+=1
    return data,i

def getPolynomials(maxDegree, numFeatures):
    features = range(0,numFeatures)
    finalList = []
    # Only get polynomials from 2(Quadractics) to the max degree
    for i in range(2,maxDegree+1):
        finalList.append(list(combinations_with_replacement(features,i)))
    finalList = [item for sublist in finalList for item in sublist]
    print(finalList)
    return finalList

def polynomialList(data,numFeatures,maxDegree):
    polynomials = getPolynomials(maxDegree,numFeatures)
    for poly in polynomials:
        newCol = np.ones((len(data),1))
        
        for idx in poly:
            newCol = np.multiply(newCol,np.reshape(data[:,idx],(len(data),1)))   
        data = np.concatenate((data,newCol),axis=1)
    return data  
        

lr = 0.5
#Read in the data
# F = Number of features
# N = Number of examples
# T = Number of test examples
F,N = sys.stdin.readline().split(" ")
F,N = int(F),int(N)
data = np.zeros((N,F))
y = np.zeros((N,1))

# Split data into features and price
for i in range(0,N):
    inputLine = tuple(map(float,sys.stdin.readline().split(" ")))
    data[i] = inputLine[:-1]
    y[i] = inputLine[-1]

# Read in Test Data
T = int(sys.stdin.readline())
testData = np.zeros((T,F))
for i in range(0,T):
    testData[i] = tuple(map(float,sys.stdin.readline().split(" ")))


# Add Polynomial features
X = polynomialList(data,F,4)  
# Add Bias Term
X = np.concatenate((np.ones((N,1)),X),axis=1)

# Random Initialisation of theta
theta = np.zeros((len(X[0]),1))
for i in range(0,len(X[0])):
    theta[i] = random.uniform(0,10)
# Calculate cost with current theta
costHistory = []
for i in range(0,10000):
    theta = gradientDescentCalculation(X,y,theta)
    cost = computeCost(X,y,theta)
    costHistory.append(cost)

 

# Test Data
testX = np.concatenate((np.ones((T,1)),testData),axis=1)
testX = polynomialList(testX,F,4)
output = np.dot(testX,theta)
print(output)



