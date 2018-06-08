# -*- coding: utf-8 -*-
"""
Author:Sara Li
Date:2018.6.7

PLA/Pocket binary classification
"""

import numpy as np
import math
import random


def sign(d):
    if(d <= 0):
        return -1.0
    else:
        return 1.0

def dataAnaly(data):
    size = data.shape
    x0 = np.ones(size[0]).reshape(size[0], 1)
    x1n = data[:,0:size[1]-1]
    x = np.zeros((size[0],size[1]))
    x[:,0:1]=x0
    x[:,1:]=x1n[:,:]
    y = data[:,-1]
    return x,y

#cycle PLA
# input: a(n* m+1)--x n*m ,y n*1  lr--learning rate
#output: update cycle
def PLA(training_data,lr = 1):
    xn,yn = dataAnaly(training_data)
    shape = xn.shape
    row = shape[0]
    col = shape[1]
    w = np.zeros((col)) #initialize weight
    
    error = 1
    count = 0
    while(error == 1):
        for i in range(row):
            dot = np.dot(xn[i,:].reshape(col),w)
            if(sign(dot) != yn[i]):
                w = w + lr*xn[i,:]*yn[i]
                count = count + 1
                error = 1
#                print ("cycle: {}".format(count))
                print ("w:{}".format(w))
            else:
                error = 0
    return count
#    
#cycle PLA with fixed, predefined random order visiting  
def PLA_random(training_data,test_cycles,lr = 1):
    cycle_sum = 0
    for i in range(test_cycles):
       np.random.shuffle(training_data)
       count = PLA(training_data,lr)
       print ("cycle: {}".format(count))
       cycle_sum = cycle_sum + count
    return cycle_sum/test_cycles
        
#pocket
def PLA_Pocket(training_data,test_data,updateCycle=50,lr = 1):
    xn,yn = dataAnaly(training_data)
    shape = xn.shape
    row = shape[0]
    col = shape[1]
    w = np.zeros((col)) #initialize weight 
    w1=w
    xtest,ytest = dataAnaly(test_data)
    
    
    count = 0
    errs = 0#error rate
    for j in range(xn.shape[0]):
        cdot = np.dot(xn[j,:].reshape(xn.shape[1]),w)                
        if(sign(cdot) != yn[j]):
            errs = errs + 1  
    #PLA (random access)       
    while (count <= updateCycle):
        random.seed()
        index = random.randrange(0, row, 1)
        dot = np.dot(xn[index,:].reshape(col),w1)
#        print('index:{}'.format(index))
        if(sign(dot) != yn[index]):
            w1 = w1 + lr*xn[index,:]*yn[index]
            nerrs = 0
            for j in range(xn.shape[0]):
                temp = xn[j,:].reshape(xn.shape[1])
                cdot = np.dot(temp,w1)                
                if(sign(cdot) != yn[j]):
                    nerrs = nerrs + 1         
            #pocket part
            if(errs > nerrs):
                errs = nerrs
                w=w1
            count=count+1 
#            print('cycle:{},w1={},errs={},nerrs={}'.format(count,w1,errs,nerrs))
    errs = 0
    for i in range(xtest.shape[0]):
       dot = np.dot(xtest[i,:].reshape(xtest.shape[1]),w)
       if(sign(dot) != ytest[i]):
           errs = errs + 1    
    errRate = errs/xtest.shape[0]
    return errRate
    
#PLA randomly access to data
def PLA_Pure_random(training_data,test_data,updateCycle=50,lr = 1):
    xn,yn = dataAnaly(training_data)
    shape = xn.shape
    row = shape[0]
    col = shape[1]
    w = np.zeros((col)) #initialize weight 
    
    xtest,ytest = dataAnaly(test_data)
    
    errs = 0#error rate
    count = 0

    while (count <= updateCycle):
        random.seed()
        index = random.randrange(0, row, 1)
        dot = np.dot(xn[index,:].reshape(col),w)
        if(sign(dot) != yn[index]):
            w = w + lr*xn[index,:]*yn[index] 
        count = count + 1

    for i in range(xtest.shape[0]):
       dot = np.dot(xtest[i,:].reshape(xtest.shape[1]),w)
       if(sign(dot) != ytest[i]):
           errs = errs + 1
    errRate = errs/xtest.shape[0]
    return errRate           

def PLA_Pocket_random(training_data,test_data,test_cycles,lr=1):
    cycle_sum = 0
    for i in range(test_cycles):
       errRate = PLA_Pocket(training_data,test_data,100) 
#       errRate = PLA_Pure_random(training_data,test_data) 
       print ("cycle: {},errRate:{}".format(i,errRate))
       cycle_sum = cycle_sum + errRate
    return cycle_sum/test_cycles    
    