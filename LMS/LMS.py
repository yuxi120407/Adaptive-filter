# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 21:56:07 2018

@author: Xi Yu
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt  
import math 
import textwrap
from sklearn.model_selection import train_test_split


Data = np.loadtxt('speech.txt')


X_train = Data[0:15000] 
X_test =  Data[15000:] 
#plt.plot(X_train)
#plt.title('speech')
power = np.sum(X_test**2)/3176


def Wiener_solution(filter_order,windows_size,whole_sample_num):
    window_num = int(whole_sample_num/windows_size)
    whole_weight = np.zeros([window_num,filter_order])
    weight = np.zeros(filter_order)
    for i in range(window_num):
        x_input = X_train[windows_size*i:windows_size*(i+1)]
        y_trainingdata = X_train[filter_order+windows_size*i:windows_size*(i+1)]
        x_trainingdata = np.zeros([windows_size-filter_order,filter_order])
        for j in range(windows_size-filter_order):
            x_trainingdata[j,:] = x_input[j:j+filter_order]
        weight = np.linalg.inv(x_trainingdata.T@x_trainingdata)@x_trainingdata.T@y_trainingdata
        whole_weight[i,:] = weight
    return whole_weight

def get_input_trainigdata(filter_order,whole_sample_num):
    X_trainingdata = np.zeros([whole_sample_num-filter_order,filter_order])
    Y_trainingdata = X_test[filter_order:whole_sample_num]
    for n in range(whole_sample_num-filter_order):
        X_trainingdata[n,:] = X_test[n:n+filter_order]

    return X_trainingdata,Y_trainingdata




def MSE(weight,filter_order,whole_sample_num):
    MSE = 0
    for i in range(whole_sample_num-filter_order):
        predict = X_trainingdata[i,:]@weight
        error_LMS = ((Y_trainingdata[i]-predict)**2)
        MSE = MSE+error_LMS
    error = MSE/(whole_sample_num-filter_order)
    error = error/power
    return error

def condition_number(filter_order, windows_size,whole_sample_num):
    
    window_num = int(whole_sample_num/windows_size)
    eta = np.zeros(window_num)
    for i in range(window_num):
        x_input = X_train[windows_size*i:windows_size*(i+1)]
        x_trainingdata = np.zeros([windows_size-filter_order,filter_order])
        for j in range(windows_size-filter_order):
            x_trainingdata[j,:] = x_input[j:j+filter_order]
        eigen_vals, eigen_vecs = np.linalg.eig(x_trainingdata.T@x_trainingdata)
        eigen_vals_max = np.max(eigen_vals)
        eigen_vals_min = np.min(eigen_vals)
        eta[i] = eigen_vals_max/eigen_vals_min
    return eta

def LMS(filter_order,windows_size,whole_sample_num,step_size):
    
    window_num = int(whole_sample_num/windows_size)
    whole_weight = np.zeros([window_num,filter_order])
    weight_LMS = np.zeros(filter_order)
    for i in range(window_num):
        x_input = X_train[windows_size*i:windows_size*(i+1)]
        y_trainingdata = X_train[filter_order+windows_size*i:windows_size*(i+1)]
        x_trainingdata = np.zeros([windows_size-filter_order,filter_order])
        for j in range(windows_size-filter_order):
            x_trainingdata[j,:] = x_input[j:j+filter_order]
        predict = x_trainingdata@weight_LMS
        error = y_trainingdata-predict
        weight_LMS = weight_LMS + step_size*x_trainingdata.T@error
        whole_weight[i,:] = weight_LMS
    return whole_weight
lms_weight1 = LMS(15,500,15000,0.02)
plt.plot(lms_weight1)
plt.plot(whole_weight)
plt.title('order=6 windows size=100')

eta = condition_number(15,500,15000)
plt.plot(eta)
whole_weight = Wiener_solution(15,500,15000)
X_trainingdata,Y_trainingdata = get_input_trainigdata(15,3176)
error = np.zeros(30)
for i in range(30):
    error[i] = MSE(whole_weight[i,:],15,3176)

error1 = np.zeros(30)
for i in range(30):
    error1[i] = MSE(lms_weight1[i,:],15,3176)  

#plt.plot(error1)
#p1 = plt.plot(error)
#p2 = plt.plot(error1)
#plt.title('order=6 windows size=500')
#plt.legend((p1[0],p2[0]),('wiener solution','LMS'), fontsize=10)
#plt.ylabel('NMSE',fontsize=10) 
#plt.xlabel('number of windows', fontsize=10) 
#plt.plot(eta)
    
