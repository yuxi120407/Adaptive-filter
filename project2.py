# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 17:24:39 2018

@author: Xi Yu
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt  
import math 
import time
import matplotlib.patches as mpatches
from sklearn import metrics
from sklearn.metrics import roc_auc_score
import scipy
from scipy import signal


data_noise = np.loadtxt('noise.txt')
data_test = np.loadtxt('test.txt')
data_call = np.loadtxt('train.txt')
data = data_call[96000]
for i in range(data_call.shape[0]):
    if(abs(data_call[i])>0.001):
        data = np.append(data,data_call[i])
#%%
call = data[1:]
call1 = call[0:14693]
call2 = call[14693:26386]
call3 = call[26386:40079]
call4 = call[40079:56772]
call5 = call[56772:71465]
call6 = call[71465:90358]
call7 = call[90358:99551]
call8 = call[99551:119644]
call9 = call[119644:135537]
call10 = call[135537:146937]
#%%
def loadinputdata(inputdata,filter_order):
    size = inputdata.shape[0]
    sampleNum = size-filter_order
    X = np.zeros([sampleNum,filter_order])
    for j in range(sampleNum):
        X[j,:] = inputdata[j:j+filter_order]
    y = inputdata[filter_order:]
    return X, y
        
def LMS(inputdata, filter_order,learningrate):
    X,y = loadinputdata(inputdata,filter_order)
    size = inputdata.shape[0]
    sampleNum = size-filter_order
    error = np.zeros(sampleNum)
    weight = np.zeros(filter_order)
    weight_track = np.zeros([sampleNum,filter_order])
    for i in range(sampleNum):
        error[i] = y[i] - X[i,:]@weight
        weight = weight + learningrate*error[i]*X[i,:].T
        weight_track[i,:] = weight
    return weight, weight_track    

def RLS(inputdata,filter_order,forget_factor):
    beta = 1/forget_factor
    size = inputdata.shape[0]  
    sampleNum = size-filter_order
    X,y = loadinputdata(inputdata,filter_order)
    first_train = X[0:100,:]
    var = np.var(first_train)
    P = 100*var*np.eye(filter_order, dtype=int)
    P = 0.01*np.eye(filter_order, dtype=int)
    error= np.zeros(sampleNum)
    weight_RLS =np.matrix(np.zeros(filter_order)).T
    weight_track = np.zeros([sampleNum,filter_order])
    for i in range(sampleNum):
        x = np.matrix(X[i,:])
        r = 1 + beta*(x@P@x.T)
        k = beta*(P@x.T)/r
        e = y[i] - x@weight_RLS
        weight_RLS = weight_RLS + k*e
        P = beta*P - np.multiply(k@k.T,r)
        weight_track[i,:] = np.array(weight_RLS.T)[0]
    weight_RLS = np.array(weight_RLS.T)[0]
    error = y - X@weight_RLS
    mse = np.mean(error**2)
    return weight_RLS, mse

def gaussian(x, mu, sig):
    a = (x-mu).T@(x-mu)
    return np.exp(-a / (2 * sig**2))

def kernel_Q(x,a,C):
    output = 0
    for i in range(len(a)):
        output = output + a[i]*gaussian(x,C[i,:],1)
    return output

def QKLMS(inputdata,filter_order,QKLMS_learning_rate):
    x_train, y_train = loadinputdata(inputdata,filter_order)
    trainNum = x_train.shape[0]
    QKLMS_error = np.zeros(trainNum)
    QKLMS_error[0] = y_train[0]
    Q_a = np.array([QKLMS_learning_rate*QKLMS_error[0]])
    C = np.array([x_train[0]])
    j = 1
    num_QKLMS = np.zeros(trainNum)
    num_QKLMS[0]= 1
    num_LMS = 1
    for i in range(1,trainNum):
        QKLMS_predict = kernel_Q(x_train[i],Q_a,C)
        QKLMS_error[i] = y_train[i]-QKLMS_predict
        distance = np.sum(np.sqrt((C-x_train[i])**2),axis=1)
        if (np.min(distance) < 0.5):
            index = np.argmin(distance)
            Q_a[index] = Q_a[index] + QKLMS_learning_rate*QKLMS_error[i]    
        else:
            new_Q_a = np.array([QKLMS_learning_rate*QKLMS_error[i]])
            Q_a= np.append(Q_a,new_Q_a)
            C= np.vstack((C,x_train[i]))
            j = j+1
        num_QKLMS[i] = j
        num_LMS = num_LMS +1
    return Q_a, C

def compute_error(inputdata, weight):
    filter_order = weight.shape[0]
    x_test,y_test = loadinputdata(inputdata,filter_order)
    error = y_test - x_test@weight
    mse = np.mean(error**2)
    return mse 

def compute_kernel_error(inputdata,filter_order,a,C):
    x_test,y_test = loadinputdata(inputdata,filter_order)
    sampleNum = x_test.shape[0]
    for n in range(sampleNum):
        error = abs(y_test[n] - kernel_Q(x_test[n],a,C))
        mse = np.mean(error)
    return mse
    

def decision_function_kernel(data):
    error_call = compute_kernel_error(data,5, Q_a ,C)
    error_noise = compute_kernel_error(data,5,Q_a_noise, C_noise)
    score = error_noise - error_call
    return score

def decision_function(data,weight_call,weight_noise):
    error_call = compute_error(data,weight_call)
    error_noise = compute_error(data,weight_noise)
    error = error_noise - error_call
    mse = np.mean(error)
    return mse
    
#%%
weight_call,_ =  RLS(call,5,0.8)   
weight_noise,_ =  RLS(data_noise,5,0.8) 

#Q_a, C = QKLMS(call1, 5, 0.5)
#Q_a_noise, C_noise = QKLMS(data_noise, 5, 0.5)
#%%
score = np.zeros(30)
for m in range(30):
    data = data_test[44100*m:44100*(m+1)] 
    score[m] = decision_function(data,weight_call,weight_noise)
    
true_label = np.array([0,0,1,1,1,0,1,1,1,0,1,0,0,1,1,0,0,1,1,1,0,0,0,0,1,1,0,0,0,0])
fpr, tpr, thresholds = metrics.roc_curve(true_label, score) 
area = roc_auc_score(true_label, score)

#%%   
plt.plot(fpr,tpr,'bo')
plt.plot(fpr,tpr,'b')
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')



#Y = scipy.signal.resample(call1, 2666)








    
    
    
    