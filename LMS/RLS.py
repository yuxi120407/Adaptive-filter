# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 11:42:28 2018

@author: Xi Yu
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt  
import math 
import textwrap

inputData = np.loadtxt('input.txt')
outputData = np.loadtxt('output.txt')


#inputData = np.loadtxt('input_0.3.txt')
#outputData = np.loadtxt('output_0.3.txt')
#inputData = np.loadtxt('input_1.5.txt')
#outputData = np.loadtxt('output_1.5.txt')

p = np.sum(inputData**2)

plt.plot(outputData)
plt.show
#%%
# Wiener solution
def Wiener_solution(filter_order,windows_size,whole_sample_num):
    #weight_row = filter_order
    #weight_column = int(whole_sample_num/windows_size)
    #window_num = weight_column
    #weight_sample = np.zeros([weight_row,weight_column])
    
    x_input = inputData[0:windows_size]
    #x_input = inputData[900:1000]
    y_trainingdata = outputData[(filter_order-1):windows_size]
    x_trainingdata = np.zeros([windows_size-filter_order+1,filter_order])
    for j in range(windows_size-filter_order+1):
        x_trainingdata[j,:] = x_input[j:j+filter_order]
    weight_sample = np.linalg.inv(x_trainingdata.T@x_trainingdata)@x_trainingdata.T@y_trainingdata

    return weight_sample

def WSNR(weight,filter_order):
    Weight_star = np.array([1,1,1,1,1,1,1,1,1,1])
    if filter_order>=10:
        padding = np.zeros(filter_order-10)
        Weight_star = np.append(Weight_star,padding)
    else:
        padding = np.zeros(10-filter_order)
        weight = np.append(weight,padding)
    a = Weight_star.T@Weight_star
    b = (Weight_star-weight).T@(Weight_star-weight)
    wsnr = 10*np.log10(a/b)
    return wsnr
    

weight1 = Wiener_solution(5,100,10000)
weight2 = Wiener_solution(5,500,10000)
weight3 = Wiener_solution(5,1000,10000)
wsnr1 = WSNR(weight1,5)
wsnr2 = WSNR(weight2,5)
wsnr3 = WSNR(weight3,5)    

#%%
#LMS
def get_input_trainigdata(filter_order,whole_sample_num):
    X_trainingdata = np.zeros([(whole_sample_num-filter_order+1),filter_order])
    Y_trainingdata = outputData[(filter_order-1):whole_sample_num]
    for n in range((whole_sample_num-filter_order+1)):
        X_trainingdata[n,:] = inputData[n:n+filter_order]
    return X_trainingdata,Y_trainingdata
    
#eigen_vals, eigen_vecs = np.linalg.eig(X_trainingdata.T@X_trainingdata)
#eigen_vals_max = np.max(eigen_vals)
#eta = 1/eigen_vals_max

def LMS(filter_order,windows_size,whole_sample_num,step_size):
    
    window_num = int(whole_sample_num/windows_size)
    whole_weight = np.zeros([window_num,filter_order])
    weight_LMS = np.zeros(filter_order)
    for i in range(window_num):
        x_input = inputData[windows_size*i:windows_size*(i+1)]
        y_trainingdata = outputData[(filter_order-1)+windows_size*i:windows_size*(i+1)]
        x_trainingdata = np.zeros([windows_size-filter_order+1,filter_order])
        for j in range(windows_size-filter_order+1):
            x_trainingdata[j,:] = x_input[j:j+filter_order]
        predict = x_trainingdata@weight_LMS
        error = y_trainingdata-predict
        weight_LMS = weight_LMS + step_size*x_trainingdata.T@error
        whole_weight[i,:] = weight_LMS
    return whole_weight

whole_weight1 = LMS(15,100,10000,0.0001)
whole_weight2 = LMS(15,500,10000,0.0001)
whole_weight3 = LMS(15,1000,10000,0.0001)

wsnr_LMS_1 = np.zeros(100)
wsnr_LMS_2 = np.zeros(20)
wsnr_LMS_3 = np.zeros(10)
for n in range(100):
    wsnr_LMS_1[n] = WSNR(whole_weight1[n,:],15)
for n in range(20):
    wsnr_LMS_2[n] = WSNR(whole_weight2[n,:],15)
for n in range(10):
    wsnr_LMS_3[n] = WSNR(whole_weight3[n,:],15)

#plt.plot(wsnr_LMS_1)
#plt.plot(wsnr_LMS_2)
#plt.plot(wsnr_LMS_3)
    

def MSE(weight,filter_order,whole_sample_num):
    MSE = 0
    for i in range(whole_sample_num-filter_order+1):
        predict = X_trainingdata[i,:]@weight
        error_LMS = ((Y_trainingdata[i]-predict)**2)
        MSE = MSE+error_LMS
    error = MSE/(whole_sample_num-filter_order+1)
    return error
    
#%%
#calculate the normalized MSE
X_trainingdata,Y_trainingdata = get_input_trainigdata(15,10000)
predict_LMS = X_trainingdata@whole_weight2[19,:].T
fig = plt.figure(figsize=(20,5))
ax = fig.add_subplot(*[1,2,2])
ax.plot(predict_LMS)
#predict_LMS = X_trainingdata@weight_LMS
#mse_LMS = np.zeros(100)
#for i in range(100):
    #mse_LMS[i] = MSE(whole_weight1[i,:],5,10000)
#plt.plot(mse_LMS)
#mse_wiener = MSE(weight2,5,10000)
#%%

def RLS(filter_order, whole_sample_num):
    weight = np.zeros(filter_order)
    identity_matrix = np.eye(5, dtype=int)
    x0_train = np.matrix(X_trainingdata[0,:])
    R_inverse = np.linalg.inv(x0_train.T@x0_train)
    for i in range(whole_sample_num-filter_order):
        xi_train = np.matrix(X_trainingdata[i+1,:])
        a = R_inverse@xi_train.T@xi_train@R_inverse
        b = identity_matrix+xi_train@R_inverse@xi_train.T
        minus_term = a@np.linalg.inv(b)
        R_inverse = R_inverse- minus_term
        error = Y_trainingdata
        weight = weight + R_inverse@xi_train.T

        
    





 









