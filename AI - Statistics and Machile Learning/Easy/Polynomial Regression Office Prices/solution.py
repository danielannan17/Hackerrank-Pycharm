import sys
import pandas as pd
import numpy as np
import random
import math
from itertools import combinations_with_replacement
import sklearn.preprocessing as pp

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

def computeCost(X,y,theta):
    hypothesis = np.dot(X,theta)
    err = np.subtract(hypothesis,y)
    errSqrd = np.dot(np.transpose(err),err)
    cost = np.multiply(0.5/N,errSqrd)
    return cost
def addPolynomialFeatures(data,*args):
    i = 0
    for feature,power in args:
        data = np.concatenate((data,data[:,[feature]]**power), axis=1)
        i+=1
    return data,i


def polynomialList(data,numFeatures,maxDegree):
    features = range(0,numFeatures)
    finalList = []
    for i in range(2,maxDegree+1):
        finalList.append(list(combinations_with_replacement(features,i)))
    finalList = [item for sublist in finalList for item in sublist]
    return finalLists

lr = 0.1
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
bestCost = 9999999
bestTheta =[]
bestFeatures = []
k = 0
poly = pp.PolynomialFeatures(4,include_bias = True)
X = poly.fit_transform(data)
F = len(X[0])
    

# Add Bias Term
#X = np.concatenate((np.ones((N,1)),X),axis=1)

# Random Initialisation of theta
theta = np.zeros((F,1))
for i in range(0,F):
    theta[i] = random.uniform(0,10)
# Calculate cost with current theta
costHistory = []
for i in range(0,40000):
    theta = gradientDescentCalculation(X,y,theta)
    cost = computeCost(X,y,theta)
    costHistory.append(cost)

if cost < bestCost:
    bestCost = cost
    bestTheta = theta
 

# Test Data
testX = poly.fit_transform(testData)
output = np.dot(testX,bestTheta)
for i in output:
    print(i[0])


