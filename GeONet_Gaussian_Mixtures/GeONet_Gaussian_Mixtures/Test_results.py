import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.autograd import Variable
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import ot
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# Declare bivariate normal density value function
def h(x, y, mu, Sigma):
    YY = np.array([[x,y]])
    XX = YY.squeeze()
    return (1/(2*math.pi))*((np.linalg.det(Sigma))**(-1/2))*(math.e)\
    **(-(1/2)*(np.matmul(np.matmul((XX - mu),np.linalg.inv(Sigma)),np.transpose(XX - mu))))


N = 30  # Mesh size of input densities, 30x30
N_2 = 40 # Geodesic mesh size


# Define function to get test results
def test_results(Cty_branch0, Cty_branch1, Cty_trunk, HJ_branch0, HJ_branch1, HJ_trunk):

    # Create mesh
    x = np.linspace(0, 5, N)
    y = np.linspace(0, 5, N)
    X, Y = np.meshgrid(x, y)

    # Define number of mixtures
    num_u0_mixtures = 5
    num_u1_mixtures = 2
    k = num_u0_mixtures + num_u1_mixtures

    

    # Construct means, variance, and covariances
    means = np.random.uniform(low=1.2, high=3.8, size=(k,2))
    variances = np.random.uniform(low=0.35, high=0.75, size=(k,2))
    covariances = np.random.uniform(low=-0.2, high=0.2, size=(k,2))

    # Add variances/covariances to arrays
    Sigma = np.zeros(shape=(2,2,k))
    for i in np.arange(0,k):
        Sigma[0,0,i] = variances[i,0]; Sigma[1,1,i] = variances[i,1]
        Sigma[0,1,i] = covariances[i,0]; Sigma[1,0,i] = covariances[i,1]
        


    # Generate coefficients of the mixtures
    c1 = c2 = c3 = c4 = c5 = 0.2
    c6 = np.random.uniform(0,0.7,1); c7=1-c6


    # Initialize arrays 
    u0_test = np.zeros(shape=(1,N,N))
    u1_test = np.zeros(shape=(1,N,N))

    u0_test = torch.zeros(size=(N_2*N_2, N*N))
    u1_test = torch.zeros(size=(N_2*N_2, N*N))

    u0_base = np.zeros(shape=(N,N))
    u1_base = np.zeros(shape=(N,N))
    u0_base_vector = np.zeros(shape=(N*N,1))
    u1_base_vector = np.zeros(shape=(N*N,1))

    # Construct Gaussian mixtures
    z = 0     
    for i in range(N):
        for j in range(N):
            u0_base[i,j] = c1*h(X[i,j], Y[i,j], means[0,:], Sigma[:,:,0])  + c2*h(X[i,j], Y[i,j], means[1,:], Sigma[:,:,1]) + \
                          c3*h(X[i,j], Y[i,j], means[2,:], Sigma[:,:,2])  + c4*h(X[i,j], Y[i,j], means[3,:], Sigma[:,:,3]) + \
                          c5*h(X[i,j], Y[i,j], means[4,:], Sigma[:,:,4])  

            u1_base[i,j] = c6*h(X[i,j], Y[i,j], means[5,:], Sigma[:,:,5])  + c7*h(X[i,j], Y[i,j], means[6,:], Sigma[:,:,6])
 
            u0_base_vector[z,0] = u0_base[i,j]
            u1_base_vector[z,0] = u1_base[i,j]
            z = z + 1
  
    # Store the initial conditions as repeated vectors for DeepONet input
    for w in range(N_2*N_2):
        u0_test[w,0:N*N] = Variable(torch.from_numpy(u0_base_vector[:,0]).float(), requires_grad=False).to(device)
        u1_test[w,0:N*N] = Variable(torch.from_numpy(u1_base_vector[:,0]).float(), requires_grad=False).to(device)
   

    # Create new mesh for the geodesic, with geodesic mesh size N_2
    x = np.linspace(0, 5, N_2)
    y = np.linspace(0, 5, N_2)
    X_new0, Y_new0 = np.meshgrid(x, y)

    u0_highres = np.zeros(shape=(N_2,N_2))
    u1_highres = np.zeros(shape=(N_2,N_2))
    X_new = np.zeros(shape=(N_2*N_2,1))
    Y_new = np.zeros(shape=(N_2*N_2,1))

    # Create initial conditions with higher-resolution mesh size
    z = 0
    for i in range(N_2):
        for j in range(N_2):
            u0_highres[i,j] = c1*h(X_new0[i,j], Y_new0[i,j], means[0,:], Sigma[:,:,0])  + c2*h(X_new0[i,j], Y_new0[i,j], means[1,:], Sigma[:,:,1]) + \
                      c3*h(X_new0[i,j], Y_new0[i,j], means[2,:], Sigma[:,:,2])  + c4*h(X_new0[i,j], Y_new0[i,j], means[3,:], Sigma[:,:,3]) + \
                      c5*h(X_new0[i,j], Y_new0[i,j], means[4,:], Sigma[:,:,4])  
 

            u1_highres[i,j] = c6*h(X_new0[i,j], Y_new0[i,j], means[5,:], Sigma[:,:,5])  + c7*h(X_new0[i,j], Y_new0[i,j], means[6,:], Sigma[:,:,6])

            X_new[z,0] = X_new0[i,j]
            Y_new[z,0] = Y_new0[i,j]
            z = z + 1
 
    X_vector_new = Variable(torch.from_numpy(X_new).float(), requires_grad=True).to(device)
    Y_vector_new = Variable(torch.from_numpy(Y_new).float(), requires_grad=True).to(device)


    # Create times for geodesics
    t_test = np.ones(shape=(N_2*N_2,5))
    t_test[:,0] = 0*t_test[:,0]; t_test[:,1] = 0.25*t_test[:,1]; t_test[:,2] = 0.5*t_test[:,2]; t_test[:,3] = 0.75*t_test[:,3]
    t_test = Variable(torch.from_numpy(t_test).float(), requires_grad=False).to(device)
    
    # Save initial conditions to device
    u0_test = Variable(u0_test.float(), requires_grad=False).to(device)
    u1_test = Variable(u1_test.float(), requires_grad=False).to(device)


    # Feed in initial conditions to continuity branches
    u_branch0 = Cty_branch0(u0_test)
    u_branch1 = Cty_branch1(u1_test)

    # Compute trunk networks
    u_trunk0 = Cty_trunk(X_vector_new, Y_vector_new, t_test[:,0].reshape(-1,1))
    u_trunk25 = Cty_trunk(X_vector_new, Y_vector_new, t_test[:,1].reshape(-1,1))
    u_trunk5 = Cty_trunk(X_vector_new, Y_vector_new, t_test[:,2].reshape(-1,1))
    u_trunk75 = Cty_trunk(X_vector_new, Y_vector_new, t_test[:,3].reshape(-1,1))
    u_trunk1 = Cty_trunk(X_vector_new, Y_vector_new, t_test[:,4].reshape(-1,1))

    # Compute geodesic at certain times
    u_test_vector0 = torch.sum( (u_branch0*u_branch1) * u_trunk0, dim=-1).unsqueeze(1)
    u_test_vector25 = torch.sum( (u_branch0*u_branch1) * u_trunk25, dim=-1).unsqueeze(1)
    u_test_vector5 = torch.sum( (u_branch0*u_branch1) * u_trunk5, dim=-1).unsqueeze(1)
    u_test_vector75 = torch.sum( (u_branch0*u_branch1) * u_trunk75, dim=-1).unsqueeze(1)
    u_test_vector1 = torch.sum( (u_branch0*u_branch1) * u_trunk1, dim=-1).unsqueeze(1)

    # Reform tensors in arrays
    u_test0 = torch.reshape(u_test_vector0, (N_2,N_2))
    u_test25 = torch.reshape(u_test_vector25, (N_2,N_2))
    u_test5 = torch.reshape(u_test_vector5, (N_2,N_2))
    u_test75 = torch.reshape(u_test_vector75, (N_2,N_2))
    u_test1 = torch.reshape(u_test_vector1, (N_2,N_2))
    
    
    
    # Plot the results
    cmap = LinearSegmentedColormap.from_list('Colormap', [ 'blue', 'mediumturquoise', 'orange', 'yellow' ])

    # Plot the geodesics at the five tmimes
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5, figsize=(12,6)) 

    ax1 = sns.heatmap(u_test0.cpu().detach().numpy(), ax=ax1, cmap=cmap, square=True, cbar=False)
    cf1 = ax1.contour(u_test0.cpu().detach().numpy(), cmap=plt.cm.rainbow)
    ax1.set(xticklabels=[]); ax1.set(yticklabels=[]); ax1.tick_params(left=False, bottom=False)

    ax2 = sns.heatmap(u_test25.cpu().detach().numpy(), ax=ax2, cmap=cmap, square=True, cbar=False)
    cf2 = ax2.contour(u_test25.cpu().detach().numpy(), cmap=plt.cm.rainbow)
    ax2.set(xticklabels=[]); ax2.set(yticklabels=[]); ax2.tick_params(left=False, bottom=False)

    ax3 = sns.heatmap(u_test5.cpu().detach().numpy(), ax=ax3, cmap=cmap, square=True, cbar=False)
    cf3 = ax3.contour(u_test5.cpu().detach().numpy(), cmap=plt.cm.rainbow)
    ax3.set(xticklabels=[]); ax3.set(yticklabels=[]); ax3.tick_params(left=False, bottom=False)

    ax4 = sns.heatmap(u_test75.cpu().detach().numpy(), ax=ax4, cmap=cmap, square=True, cbar=False)
    cf4 = ax4.contour(u_test75.cpu().detach().numpy(), cmap=plt.cm.rainbow)
    ax4.set(xticklabels=[]); ax4.set(yticklabels=[]); ax4.tick_params(left=False, bottom=False)

    ax5 = sns.heatmap(u_test1.cpu().detach().numpy(), ax=ax5, cmap=cmap, square=True, cbar=False)
    cf5 = ax5.contour(u_test1.cpu().detach().numpy(), cmap=plt.cm.rainbow)
    ax5.set(xticklabels=[]); ax5.set(yticklabels=[]); ax5.tick_params(left=False, bottom=False)

    plt.xticks([])
    plt.yticks([])

    print("GeONet results:")
    plt.show()
    
    
    ###############  Run POT algorithm  ########
    ############################################
    
    f1 = np.zeros(shape=(N_2,N_2)); f2 = np.zeros(shape=(N_2,N_2)); f3 = np.zeros(shape=(N_2,N_2)); f4 = np.zeros(shape=(N_2,N_2))
    for i in np.arange(N_2):
        for j in np.arange(N_2):
            f1[i,j] = u0_highres[i,j]; f2[i,j] = u1_highres[i,j]; f3[i,j] = 1; f4[i,j] = 1

            
    f1 = ((N_2/5)**2)*f1 / np.sum(f1); f2 = ((N_2/5)**2)*f2 / np.sum(f2)
    f3 = ((N_2/5)**2)*f3 / np.sum(f3); f4 = ((N_2/5)**2)*f4 / np.sum(f4)
    A = np.array([f1, f2, f3, f4])
    nb_images = 5
    v1 = np.array((1, 0, 0, 0)); v2 = np.array((0, 1, 0, 0)); v3 = np.array((0, 0, 1, 0)); v4 = np.array((0, 0, 0, 1))
    OT = np.zeros(shape=(nb_images,nb_images,N_2,N_2))


    # regularization parameter
    reg = 0.004

    for i in range(nb_images):
        for j in range(nb_images):
            tx = float(i) / (nb_images - 1); ty = float(j) / (nb_images - 1)
            tmp1 = (1 - tx) * v1 + tx * v2; tmp2 = (1 - tx) * v3 + tx * v4; weights = (1 - ty) * tmp1 + ty * tmp2

            if i == 0 and j == 0:
                OT[i,j,:,:] = f1
            elif i == 0 and j == (nb_images - 1):
                OT[i,j,:,:] = f3
            elif i == (nb_images - 1) and j == 0:
                OT[i,j,:,:] = f2
            elif i == (nb_images - 1) and j == (nb_images - 1):
                OT[i,j,:,:] = f4
            else:
                OT[i,j,:,:] = ot.bregman.convolutional_barycenter2d(A, reg, weights)
                
        
    # Plot the POT results
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5, figsize=(12,6)) 

    ax1 = sns.heatmap(OT[0,0,:,:], ax=ax1, cmap=cmap, square=True, cbar=False)
    cf1 = ax1.contour(OT[0,0,:,:], cmap=plt.cm.rainbow)
    ax1.set(xticklabels=[]); ax1.set(yticklabels=[]); ax1.tick_params(left=False, bottom=False)

    ax2 = sns.heatmap(OT[1,0,:,:], ax=ax2, cmap=cmap, square=True, cbar=False)
    cf2 = ax2.contour(OT[1,0,:,:], cmap=plt.cm.rainbow)
    ax2.set(xticklabels=[]); ax2.set(yticklabels=[]); ax2.tick_params(left=False, bottom=False)

    ax3 = sns.heatmap(OT[2,0,:,:], ax=ax3, cmap=cmap, square=True, cbar=False)
    cf3 = ax3.contour(OT[2,0,:,:], cmap=plt.cm.rainbow)
    ax3.set(xticklabels=[]); ax3.set(yticklabels=[]); ax3.tick_params(left=False, bottom=False)

    ax4 = sns.heatmap(OT[3,0,:,:], ax=ax4, cmap=cmap, square=True, cbar=False)
    cf4 = ax4.contour(OT[3,0,:,:], cmap=plt.cm.rainbow)
    ax4.set(xticklabels=[]); ax4.set(yticklabels=[]); ax4.tick_params(left=False, bottom=False)

    ax5 = sns.heatmap(OT[4,0,:,:], ax=ax5, cmap=cmap, square=True, cbar=False)
    cf5 = ax5.contour(OT[4,0,:,:], cmap=plt.cm.rainbow)
    ax5.set(xticklabels=[]); ax5.set(yticklabels=[]); ax5.tick_params(left=False, bottom=False)

    plt.xticks([])
    plt.yticks([])

    print("Reference results:")
    plt.show()
    
    
