# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 20:57:15 2018

@author: Xi Yu
"""


import numpy as np
import matplotlib
import matplotlib.pyplot as plt  
import math 
import time
import matplotlib.patches as mpatches
#%%

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
trainNum = 4500
testNum = 500
filter_order = 5
learning_rate = 0.001
error = np.zeros(trainNum)
test_MSE = np.zeros(trainNum)
nmse_LMS = np.zeros(trainNum)
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
X = np.zeros([sampleNum,filter_order])
weight = np.zeros(filter_order)
weight_time = np.zeros([trainNum,filter_order])
#mse = np.zeros(Numsample)
zeros = np.zeros(filter_order-1)
x = np.append(zeros,x)
for j in range(sampleNum):
    X[j,:] = x[j:j+filter_order]
x_train = X[0:trainNum]
x_test = X[trainNum:]
y_train = desired[0:trainNum]
y_test = desired[trainNum:]
power_test = np.sum(x_test**2)/testNum
#%%


std_x = np.std(x, ddof = 1)
std_opt = std_x*((4/5000)*(1/11))**0.1111
#%%LMS
for i in range(trainNum):
     error[i] = y_train[i] - x_train[i,:]@weight
     weight = weight + learning_rate*error[i]*x_train[i,:].T
     weight_time[i,:] = weight
     test_MSE[i] = ((y_test-x_test@weight).T@(y_test-x_test@weight))/testNum
     nmse_LMS[i] = test_MSE[i]/power_test
#%%
a = error**2
plot_error = a[0:4500:45]
plt.plot(plot_error)
#%%KLMS
def kernel(x,a):
    output = 0
    for i in range(len(a)):
        output = output + a[i]*gaussian(x,X[i],10)
    return output
klms_learningrate = 0.6
klms_error = np.zeros(trainNum)
klms_error[0] = y_train[0]
a = np.array([klms_learningrate*klms_error[0]])
for i in range(1,trainNum):
    predict = kernel(x_train[i],a)
    klms_error[i] = y_train[i]-predict
    new_a = np.array([klms_learningrate*klms_error[i]])
    a= np.vstack((a,new_a))

#%%QKLMS
def kernel_Q(x,a,C):
    output = 0
    for i in range(len(a)):
        output = output + a[i]*gaussian(x,C[i,:],1)
    return output
QKLMS_error = np.zeros(trainNum)
QKLMS_learning_rate = 0.6
QKLMS_error[0] = y_train[0]
Q_a = np.array([learning_rate*QKLMS_error[0]])
C = np.array([x_train[0]])
j = 1
num_QKLMS = np.zeros(trainNum)
num_QKLMS[0]= 1
num_LMS = 1
m_LMS = 0
nmse_QKLMS= np.zeros(100)
for i in range(1,trainNum):
    QKLMS_predict = kernel_Q(x_train[i],Q_a,C)
    QKLMS_error[i] = y_train[i]-QKLMS_predict
    distance = np.sum(np.sqrt((C-x_train[i])**2),axis=1)
    if (np.min(distance) < 3):
        index = np.argmin(distance)
        Q_a[index] = Q_a[index] + QKLMS_learning_rate*QKLMS_error[i]    
    else:
        new_Q_a = np.array([QKLMS_learning_rate*QKLMS_error[i]])
        Q_a= np.append(Q_a,new_Q_a)
        C= np.vstack((C,x_train[i]))
        j = j+1
    num_QKLMS[i] = j
    num_LMS = num_LMS +1
    if(num_LMS%45==0):
        QKLMS_y_predict = np.zeros(testNum)
        for n in range(testNum):
            QKLMS_y_predict[n] = kernel_Q(x_test[n],Q_a,C)
        QKLMS_test_error = y_test-QKLMS_y_predict
        nmse_QKLMS[m_LMS] = np.mean(QKLMS_test_error**2)/power_test
        m_LMS = m_LMS+1
#plt.plot(QKMCC_error**2)
plt.plot(nmse_QKLMS,'b^')
plt.plot(nmse_QKLMS,'b') 
print(nmse_QKLMS[99])
#%%QKMCC
def kernel_Q_MCC(x,a,C):
    output = 0
    for i in range(len(a)):
        output = output + a[i]*gaussian(x,C[i,:],3)
    return output

QKMCC_error = np.zeros(trainNum)
QKMCC_learning_rate = 0.5
MCC_sigma = 3
QKMCC_error[0] = y_train[0]
Q_MCC_a = np.array([learning_rate*QKMCC_error[0]])
MCC_C = np.array([x_train[0]])
MCC_j = 1
num_MCC = 1
m_MCC = 0
num_QKMCC = np.zeros(trainNum)
num_QKMCC[0]= 1
nmse_QKMCC= np.zeros(100)
for i in range(1,trainNum):
    QKMCC_predict = kernel_Q_MCC(x_train[i],Q_MCC_a,MCC_C)
    QKMCC_error[i] = y_train[i]-QKMCC_predict
    distance = np.sum(np.sqrt((MCC_C-x_train[i])**2),axis=1)
    if (np.min(distance) < 3):
        index = np.argmin(distance)
        Q_MCC_a[index] = Q_MCC_a[index] + QKMCC_learning_rate*QKMCC_error[i]*(np.exp((-1)*QKMCC_error[i]**2/2*MCC_sigma**2))    
    else:
        new_Q_MCC_a = np.array([QKMCC_learning_rate*QKMCC_error[i]*(np.exp((-1)*QKMCC_error[i]**2/2*MCC_sigma**2))])
        Q_MCC_a= np.append(Q_MCC_a,new_Q_MCC_a)
        MCC_C= np.vstack((MCC_C,x_train[i]))
        MCC_j = MCC_j+1
    num_QKMCC[i] = MCC_j
    num_MCC = num_MCC +1
    if(num_MCC%45==0):
        QKMCC_y_predict = np.zeros(testNum)
        for n in range(testNum):
            QKMCC_y_predict[n] = kernel_Q_MCC(x_test[n],Q_MCC_a,MCC_C)
        QKMCC_test_error = y_test-QKMCC_y_predict
        nmse_QKMCC[m_MCC] = np.mean(QKMCC_test_error**2)/power_test
        m_MCC = m_MCC+1
#plt.plot(QKMCC_error**2)
plt.plot(nmse_QKMCC,'b^')
plt.plot(nmse_QKMCC,'b')   
print(nmse_QKMCC[99])
#%% MEE
kernel_size_Gaussian = 3
kernel_size_MEE = 3
def kernel_Q_MEE(x,a,C):
    output = 0
    for i in range(len(a)):
        #output = output + a[i]*(gaussian(x1,C[i,:],kernel_size_MEE)-gaussian(x,C[i,:],kernel_size_MEE))
        output = output + a[i]*gaussian(x,C[i,:],kernel_size_MEE)
    return output

def gaussian_prime(error1,error2,sigma):
    error = error1-error2
    gaussian_term = np.exp(-error**2 / (2 * sigma**2))
    result = (1/sigma**2)*gaussian_term*error
    return result
        
QKMEE_error = np.zeros(trainNum)
QKMEE_learning_rate = 1
QKMEE_error[0] = y_train[0]
Q_MEE_a = np.array([QKMEE_learning_rate*QKMEE_error[0]])
MEE_C = np.array([x_train[0]])
MEE_j = 1
num_QKMEE = np.zeros(trainNum)
num_QKMEE[0]= 1
nmse_QKMEE3= np.zeros(100)
m = 0
num = 1
mean = np.mean(y_train)
for i in range(1,trainNum):
    QKMEE_predict = kernel_Q_MEE(x_train[i],Q_MEE_a,MEE_C)
    QKMEE_predict_previous = kernel_Q_MEE(x_train[i-1],Q_MEE_a,MEE_C)
    QKMEE_error[i] = y_train[i]-QKMEE_predict-mean
    QKMEE_error_previous = y_train[i-1]-QKMEE_predict_previous-mean
    distance = np.sum(np.sqrt((MEE_C-x_train[i])**2),axis=1)
    if (np.min(distance) < 3):
        index = np.argmin(distance)
        Q_MEE_a[index] = Q_MEE_a[index] + QKMEE_learning_rate*gaussian_prime(QKMEE_error[i],QKMEE_error_previous,kernel_size_Gaussian)   
    else:
        new_Q_MEE_a = np.array([QKMEE_learning_rate*gaussian_prime(QKMEE_error[i],QKMEE_error_previous,kernel_size_Gaussian)])
        Q_MEE_a= np.append(Q_MEE_a,new_Q_MEE_a)
        MEE_C= np.vstack((MEE_C,x_train[i]))
        MEE_j = MEE_j+1
    num_QKMEE[i] = MEE_j
    num = num +1
    if(num%45==0):
        QKMEE_y_predict = np.zeros(testNum)
        for n in range(testNum):
            QKMEE_y_predict[n] = kernel_Q_MEE(x_test[n],Q_MEE_a,MEE_C)
        QKMEE_test_error = y_test-QKMEE_y_predict-mean
        nmse_QKMEE3[m] = np.mean(QKMEE_test_error**2)/power_test
        m = m+1
#%%
iteration = np.arange(0,4500,45)
plt.plot(iteration,nmse_QKMEE3,'r^')
plt.plot(iteration,nmse_QKMEE3,'r')
plt.ylabel('testing NMSE',fontsize=10) 
plt.xlabel('iteration', fontsize=10) 
plt.title('learning curve')
print(nmse_QKMEE3[99])

#%%
iteration = np.arange(0,4500,45)
p1 = plt.plot(iteration,nmse_QKMEE,'c^')
p1 = plt.plot(iteration,nmse_QKMEE,'c')
p2 = plt.plot(iteration,nmse_QKMEE1,'b^')
p2 = plt.plot(iteration,nmse_QKMEE1,'b')
p3 = plt.plot(iteration,nmse_QKMEE2,'g^')
p3 = plt.plot(iteration,nmse_QKMEE2,'g')
p4 = plt.plot(iteration,nmse_QKMEE3,'y^')
p4 = plt.plot(iteration,nmse_QKMEE3,'y')
p5 = plt.plot(iteration,nmse_QKMEE4,'r^')
p5 = plt.plot(iteration,nmse_QKMEE4,'r')
plt.legend((p1[0],p2[0],p3[0],p4[0],p5[0]),('sigma1=6,sigma2=8','sigma1=8,sigma2=8', 'sigma1=10,sigma2=10','sigma1=1,sigma2=3','sigma1=3,sigma2=3' ), fontsize=10)
plt.ylabel('testing NMSE',fontsize=10) 
plt.xlabel('iteration', fontsize=10) 
plt.title('learning curve')
print(nmse_QKMEE2[99])
#plt.plot(QKMEE_error**2)
#%%KMEE
def kernel_KMEE(x,a):
    output = 0
    for i in range(len(a)):
        output = output + a[i]*gaussian(x,x_train[i,:],6)
    return output

def gaussian_prime(error1,error2,sigma):
    error = error1-error2
    gaussian_term = np.exp(-error**2 / (2 * sigma**2))
    result = (1/sigma**2)*gaussian_term*error
    return result

mean = np.mean(desired)
KMEE_learning_rate = 0.6
KMEE_error = np.zeros(trainNum)
KMEE_error[0] = y_train[0]
KMEE_a = np.array([KMEE_learning_rate*KMEE_error[0]])

for i in range(1,trainNum):
    KMEE_predict = kernel_KMEE(x_train[i],KMEE_a)
    KMEE_predict_previous = kernel_KMEE(x_train[i-1],KMEE_a)
    KMEE_previous_error = y_train[i-1]-KMEE_predict_previous-mean
    KMEE_error[i] = y_train[i]-KMEE_predict-mean
    new_KMEE_a = np.array([KMEE_learning_rate*gaussian_prime(KMEE_error[i],KMEE_previous_error,3)])
    KMEE_a= np.vstack((KMEE_a,new_KMEE_a))
plt.plot(KMEE_error**2)
#%%
iteration = np.arange(0,4500,45)
p1 = plt.plot(iteration,nmse_QKMEE4,'r^')
p1 = plt.plot(iteration,nmse_QKMEE4,'r')
p2 = plt.plot(iteration, nmse_QKMCC,'b^')
p2 = plt.plot(iteration, nmse_QKMCC,'b')
p3 = plt.plot(iteration,nmse_QKLMS,'c^')
p3 = plt.plot(iteration,nmse_QKLMS,'c')
plt.legend((p1[0],p2[0],p3[0]),('KMEE','KMCC', 'KLMS'), fontsize=10)
plt.ylabel('testing NMSE',fontsize=10) 
plt.xlabel('iteration', fontsize=10) 
plt.title('learning curve with different algorithm')
#%%
#%%
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
#p1 = plt.plot(y_test,'r')
p1 = plt.hist(QKMEE_test_error,color='green')
p2 = plt.hist(QKLMS_test_error,color='blue')
p3 = plt.hist(QKMCC_test_error,color='red')
#plt.legend((p1[0],p2[0],p3[0],p4[0]),('LMS' , 'MCC', 'QKLMS', 'QKLMCC'), fontsize=10)

plt.xlabel('errors', fontsize=10) 
plt.ylabel('number of errors', fontsize=10) 
ax.set_title('histogram of errors in test dataset') 

green_patch = mpatches.Patch(color='green', label='QKMCC')
blue_patch = mpatches.Patch(color='blue', label='QKLMS')
red_patch = mpatches.Patch(color='red', label='QKMEE')
plt.legend(handles=[green_patch,blue_patch,red_patch])
plt.show()