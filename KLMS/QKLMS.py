# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 10:19:05 2018

@author: Xi Yu
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt  
import math 
import time

def wgn(x, snr):
    snr = 10**(snr/10.0)
    xpower = np.sum(x**2)/len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)

def gaussian(x, mu, sig):
    a = (x-mu).T@(x-mu)
    return np.exp(-a / (2 * sig**2))

mu = 0
sigma = 1
sampleNum = 5000
y = np.random.normal(mu, sigma, sampleNum )
t = np.zeros(5000)
q = np.zeros(5000)
t[0] = -0.8*y[0]
for i in range(1,5000):
    t[i] = -0.8*y[i] + 0.7*y[i-1]
for j in range(5000):
    q[j] = t[j] + 0.25*t[j]**2 + 0.11*t[j]**3


lms_start_time = time.time()
def LMS(filter_order, Numsample,learning_rate):
    all_error = np.zeros([2000,5000])
    for n in range(2000):
        mu = 0
        sigma = 1
        sampleNum = 5000
        y = np.random.normal(mu, sigma, sampleNum )
        t = np.zeros(5000)
        q = np.zeros(5000)
        t[0] = -0.8*y[0]
        for i in range(1,5000):
            t[i] = -0.8*y[i] + 0.7*y[i-1]
        for j in range(5000):
            q[j] = t[j] + 0.25*t[j]**2 + 0.11*t[j]**3
        noise = wgn(q,15)
        x = q + noise
        y_delay = np.array([0,0])
        y = np.append(y_delay, y)
        desired = y[0:5000]
        X = np.zeros([Numsample,filter_order])
        weight = np.zeros(filter_order)
        weight_time = np.zeros([Numsample,filter_order])
        #mse = np.zeros(Numsample)
        zeros = np.zeros(filter_order-1)
        x = np.append(zeros,x)
        for j in range(Numsample):
            X[j,:] = x[j:j+filter_order]
        for i in range(Numsample):
             error[i] = desired[i] - X[i,:]@weight
             weight = weight + learning_rate*error[i]*X[i,:].T
             weight_time[i,:] = weight
         #mse[i] = (0.5*(desired-X@weight).T@(desired-X@weight))/Numsample
         #mse[i] = 0.5*np.sum(error**2)/(i+1)
        all_error[n,:]= error**2
    f0 = np.mean(all_error,axis=0)
    
    return weight_time, weight, f0
#%%
error = np.zeros(5000)
weight_time, weight, mse = LMS(5,5000,0.001)
print("--- %s seconds ---" % (time.time() - lms_start_time))
#%%
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(121)
ax.plot(y)
plt.xlabel('number of samples', fontsize=10) 
ax.set_title('input signal')
ax = fig.add_subplot(122)
#ax.plot(x)
plt.xlabel('number of samples', fontsize=10) 
ax.set_title('observed signal')
plt.show
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(121)
ax.plot(weight_time)
plt.xlabel('number of samples', fontsize=10) 
ax.set_title('weight track')
ax = fig.add_subplot(122)
ax.plot(mse)
plt.xlabel('number of samples', fontsize=10) 
plt.ylabel('MSE', fontsize=10)
ax.set_title('Learning curve(LMS)') 

#%%
def kernel(x,a):
    output = 0
    for i in range(len(a)):
        output = output + a[i]*gaussian(x,X[i],10)
    return output
kernel_size = 1
learning_rate = 0.6
klms_start_time = time.time()
all_error = np.zeros([20,5000])

for n in range(20):
    y = np.random.normal(0, 1, 5000)
    t = np.zeros(5000)
    q = np.zeros(5000)
    t[0] = -0.8*y[0]
    for i in range(1,5000):
        t[i] = -0.8*y[i] + 0.7*y[i-1]
    for j in range(5000):
        q[j] = t[j] + 0.25*t[j]**2 + 0.11*t[j]**3
    noise = wgn(q,15)
    x = q + noise
    y_delay = np.array([0,0])
    y = np.append(y_delay, y)
    desired = y[0:5000]
    X = np.zeros([5000,5])
    zeros = np.zeros(4)
    x = np.append(zeros,x)
    for j in range(5000):
        X[j,:] = x[j:j+5]
    ##KLMS 
    error[0] = desired[0]
    a = np.array([learning_rate*error[0]])
    mse_kernel = np.zeros(5000)
    for i in range(1,5000):
        predict = kernel(X[i],a)
        error[i] = desired[i]-predict
        new_a = np.array([learning_rate*error[i]])
        a= np.vstack((a,new_a))
    all_error[n,:]= error**2
f = np.mean(all_error,axis=0)
    #mse_kernel1[i]=0.5*np.sum(error**2)/(i+1)
print("--- %s seconds ---" % (time.time() - klms_start_time))

#%% plot the learning rate of the KLMS
fig = plt.figure()
ax = fig.add_subplot(111)
p1 = plt.plot(mse_kernel)
p2 = plt.plot(mse_kernel1)
p3 = plt.plot(mse_kernel2)
p4 = plt.plot(mse_kernel3)   
plt.legend((p1[0],p2[0],p3[0],p4[0]),('kernel size=1','kernel sizee=6', 'kernel size=10','kernel size=20' ), fontsize=10)
plt.ylabel('MSE',fontsize=10) 
plt.xlabel('number of samples', fontsize=10) 
ax.set_title('Learning curve')
plt.show()     
#%%
g = np.zeros(5000)
for n in range(5000):
    g[n] = kernel(X[n],a)
    

#%%QKLMS
#np.vstack
qklms_start_time = time.time()
def kernel_Q(x,a,C):
    output = 0
    for i in range(len(a)):
        output = output + a[i]*gaussian(x,C[i,:],10)
    return output

all_error1 = np.zeros([200,5000])
kernel_size = 6
learning_rate = 0.6
for n in range(200):
    y = np.random.normal(0, 1, 5000)
    t = np.zeros(5000)
    q = np.zeros(5000)
    t[0] = -0.8*y[0]
    for i in range(1,5000):
        t[i] = -0.8*y[i] + 0.7*y[i-1]
    for j in range(5000):
        q[j] = t[j] + 0.25*t[j]**2 + 0.11*t[j]**3
    noise = wgn(q,15)
    x = q + noise
    y_delay = np.array([0,0])
    y = np.append(y_delay, y)
    desired = y[0:5000]
    X = np.zeros([5000,5])
    zeros = np.zeros(4)
    x = np.append(zeros,x)
    for j in range(5000):
        X[j,:] = x[j:j+5]
    error = np.zeros(5000)
    error[0] = desired[0]
    a = np.array([learning_rate*error[0]])
    C = np.array([X[0]])
    mse_Qkernel4= np.zeros(5000)
    j = 0
    num_QKLMS4 = np.zeros(5000)
    for i in range(1,5000):
        predict = kernel_Q(X[i],a,C)
        error[i] = desired[i]-predict
        distance = np.sum(np.sqrt((C-X[i])**2),axis=1)
        if (np.min(distance) < 3):
            index = np.argmin(distance)
            a[index] = a[index] + learning_rate*error[i]    
        else:
            new_a = np.array([learning_rate*error[i]])
            a= np.append(a,new_a)
            C= np.vstack((C,X[i]))
            j = j+1
        num_QKLMS4[i] = j
    all_error1[n,:]= error**2
    #mse_Qkernel4[i]=0.5*np.sum(error**2)/(i+1)
f1 = np.mean(all_error1,axis=0)
print("--- %s seconds ---" % (time.time() - qklms_start_time))
#%%
fig = plt.figure()
ax = fig.add_subplot(111)
p1=plt.plot(f)
p2=plt.plot(f1)
p3=plt.plot(mse)
plt.legend((p1[0],p2[0],p3[0]),('KLMS', 'QKLMS','LMS' ), fontsize=10)
plt.ylabel('MSE',fontsize=10) 
plt.xlabel('number of samples', fontsize=10) 
ax.set_title('Learning curve')
plt.show() 
#%%plot learning curve of the QKLMS
fig = plt.figure()
ax = fig.add_subplot(111)
p1 = plt.plot(mse_Qkernel)
p2 = plt.plot(mse_Qkernel1)
p3 = plt.plot(mse_Qkernel2)
p4 = plt.plot(mse_Qkernel3) 
p5 = plt.plot(mse_Qkernel4)   
plt.legend((p1[0],p2[0],p3[0],p4[0],p5[0]),('quantization threshold=1','quantization threshold=2', 'quantization threshold=3','quantization threshold=5','quantization threshold=10'  ), fontsize=10)
plt.ylabel('MSE',fontsize=10) 
plt.xlabel('number of samples', fontsize=10) 
ax.set_title('Learning curve')
plt.show()  
#%%
fig = plt.figure()
ax = fig.add_subplot(111)
p1 = plt.plot(num_QKLMS)
p2 = plt.plot(num_QKLMS1)
p3 = plt.plot(num_QKLMS2)
p4 = plt.plot(num_QKLMS3)
p5 = plt.plot(num_QKLMS4)   
plt.legend((p1[0],p2[0],p3[0],p4[0],p5[0]),('quantization threshold=1','quantization threshold=2','quantization threshold=3', 'quantization threshold=5','quantization threshold=10' ), fontsize=10)
plt.ylabel('network size',fontsize=10) 
plt.xlabel('number of iteration', fontsize=10) 
ax.set_title('growth curve')
plt.show()     
#%%plot learning curve of LMS, KLMS and QKLMS
fig = plt.figure()
ax = fig.add_subplot(111)
p1 = plt.plot(mse)
p2 = plt.plot(mse_kernel)
p3 = plt.plot(mse_Qkernel)   
plt.legend((p1[0],p2[0],p3[0]),('LMS', 'KLMS','QKLMS' ), fontsize=10)
plt.ylabel('MSE',fontsize=10) 
plt.xlabel('number of samples', fontsize=10) 
ax.set_title('Learning curve')
plt.show() 
    
    











   
    
    
    