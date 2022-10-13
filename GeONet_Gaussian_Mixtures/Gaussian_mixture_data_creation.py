import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np
import math
import os
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Declare bivariate normal density value function
def h(x, y, mu, Sigma):
    YY = np.array([[x,y]])
    XX = YY.squeeze()
    return (1/(2*math.pi))*((np.linalg.det(Sigma))**(-1/2))*(math.e)\
    **(-(1/2)*(np.matmul(np.matmul((XX - mu),np.linalg.inv(Sigma)),np.transpose(XX - mu))))


#### Create training data

num_training_data = 20 # number of different initial conditions
N = 30 # mesh size

# Create mesh
x = np.linspace(0, 5, N)
y = np.linspace(0, 5, N)
X, Y = np.meshgrid(x, y)

# Define Gaussian mixture parameters
num_u0_mixtures = 5
num_u1_mixtures = 2
k = num_u0_mixtures + num_u1_mixtures

# Create empty grids to store training data
u0 = np.zeros(shape=(num_training_data,N,N))
u1 = np.zeros(shape=(num_training_data,N,N))
u0_vector = np.zeros(shape=(num_training_data*N*N,1))
u1_vector = np.zeros(shape=(num_training_data*N*N,1))

# Concatenation of the two initial conditions
u0 = torch.zeros(size=(num_training_data*N*N, N*N))
u1 = torch.zeros(size=(num_training_data*N*N, N*N))

X_vector = np.zeros(shape=(num_training_data*N*N,1))
Y_vector = np.zeros(shape=(num_training_data*N*N,1))
u0_vector = np.zeros(shape=(num_training_data*N*N,1))
u1_vector = np.zeros(shape=(num_training_data*N*N,1))

for num in range(num_training_data):

    ###### Declare Gaussian parameters here
    means = np.random.uniform(low=1.3, high=3.7, size=(k,2))
    variances = np.random.uniform(low=0.4, high=1.0, size=(k,2))
    covariances = np.random.uniform(low=-0.4, high=0.4, size=(k,2))
    Sigma = np.zeros(shape=(2,2,k))
    for i in np.arange(0,k):
        Sigma[0,0,i] = variances[i,0]; Sigma[1,1,i] = variances[i,1]
        Sigma[0,1,i] = covariances[i,0]; Sigma[1,0,i] = covariances[i,1]
        
    c1 = np.random.uniform(0.0, 0.9,1); c2 = np.random.uniform(0.0, 1-c1-0.05,1);
    c3 = np.random.uniform(0.0, 1-c1-c2,1); c4 = np.random.uniform(0.0, 1-c1-c2-c3,1)
    c5 =  1-c1-c2-c3-c4
    c6 = np.random.uniform(0,0.7,1); c7=1-c6

    u0_base = np.zeros(shape=(N,N))
    u1_base = np.zeros(shape=(N,N))

    u0_base_vector = np.zeros(shape=(N*N,1))
    u1_base_vector = np.zeros(shape=(N*N,1))

    z = 0
    for i in range(N):
        for j in range(N):
            u0_base[i,j] = c1*h(X[i,j], Y[i,j], means[0,:], Sigma[:,:,0])  + c2*h(X[i,j], Y[i,j], means[1,:], Sigma[:,:,1]) + \
                      c3*h(X[i,j], Y[i,j], means[2,:], Sigma[:,:,2])  + c4*h(X[i,j], Y[i,j], means[3,:], Sigma[:,:,3]) + \
                      c5*h(X[i,j], Y[i,j], means[4,:], Sigma[:,:,4])  

            u1_base[i,j] = c6*h(X[i,j], Y[i,j], means[5,:], Sigma[:,:,5])  + c7*h(X[i,j], Y[i,j], means[6,:], Sigma[:,:,6])

            X_vector[num*N*N + z,0] = X[i,j]
            Y_vector[num*N*N + z,0] = Y[i,j]
            u0_vector[num*N*N + z,0] = u0_base[i,j]
            u1_vector[num*N*N + z,0] = u1_base[i,j]
            u0_base_vector[z,0] = u0_base[i,j]
            u1_base_vector[z,0] = u1_base[i,j]
            z = z + 1
    
    # Add Gaussian mixtures together in a vector repeated N*N times
    for w in range(N*N):
        u0[N*N*num+w,0:N*N] = Variable(torch.from_numpy(u0_base_vector[:,0]).float(), requires_grad=False).to(device)
        u1[N*N*num+w,0:N*N] = Variable(torch.from_numpy(u1_base_vector[:,0]).float(), requires_grad=False).to(device)
 


# Save initial condition vectors to device
u0 = Variable(u0.float(), requires_grad=False).to(device)
u1 = Variable(u1.float(), requires_grad=False).to(device)


# Convert to torch tensors
X_vector = Variable(torch.from_numpy(X_vector).float(), requires_grad=True).to(device)
Y_vector = Variable(torch.from_numpy(Y_vector).float(), requires_grad=True).to(device)
u0_vector = Variable(torch.from_numpy(u0_vector).float(), requires_grad=True).to(device)
u1_vector = Variable(torch.from_numpy(u1_vector).float(), requires_grad=True).to(device)

# Declare vector of zeros
all_zeros = np.zeros((N*N*num_training_data,1))
pt_all_zeros = Variable(torch.from_numpy(all_zeros).float(), requires_grad=False).to(device)
