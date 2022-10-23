import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np
import random
from random import randint
from pathlib import Path
from mpl_toolkits.axes_grid1 import AxesGrid

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#### Train network

mse_cost_function = torch.nn.MSELoss() # Mean squared error


def training_algorithm(iterations, num_batches, batch_size, N, num_training_data, \
                       u0, u1, u0_vector, u1_vector, X_vector, Y_vector, pt_all_zeros, \
                       Cty, HJ, \
                       Cty_branch0, Cty_branch1, Cty_trunk, \
                       HJ_branch0, HJ_branch1, HJ_trunk, \
                       num_rows, coefficients, loss_matrix):\
    
    optimizer_HJ_branch0 = torch.optim.Adam(HJ_branch0.parameters(), lr=0.00005)
    optimizer_HJ_branch1 = torch.optim.Adam(HJ_branch1.parameters(), lr=0.00005)
    optimizer_HJ_trunk = torch.optim.Adam(HJ_trunk.parameters(), lr=0.00005)
    optimizer_Cty_branch0 = torch.optim.Adam(Cty_branch0.parameters(), lr=0.00005)
    optimizer_Cty_branch1 = torch.optim.Adam(Cty_branch1.parameters(), lr=0.00005)
    optimizer_Cty_trunk = torch.optim.Adam(Cty_trunk.parameters(), lr=0.00005)
    
    for epoch in range(iterations):

        index_list = range(0, N*N*num_training_data)
        
        coefficients[0] = np.std(loss_matrix[:,0])/np.mean(loss_matrix[:,0])
        coefficients[1] = np.std(loss_matrix[:,1])/np.mean(loss_matrix[:,1])
        coefficients[2] = np.std(loss_matrix[:,2])/np.mean(loss_matrix[:,2])
        coefficients[3] = np.std(loss_matrix[:,3])/np.mean(loss_matrix[:,3])
        coefficients = coefficients/( np.sum(coefficients) ) 
    
        # Randomly sample in bounded domain
        x_tensor = np.random.uniform(0,5,size=(N*N*num_training_data,1))
        y_tensor = np.random.uniform(0,5,size=(N*N*num_training_data,1))
        t_tensor = np.random.uniform(0,1,size=(N*N*num_training_data,1))
        t0_tensor = np.zeros((N*N*num_training_data,1))
        t1_tensor = np.ones((N*N*num_training_data,1))
    
        # Convert to tensor and save to device
        x_tensor = Variable(torch.from_numpy(x_tensor).float(), requires_grad=True).to(device)
        y_tensor = Variable(torch.from_numpy(y_tensor).float(), requires_grad=True).to(device)
        t_tensor = Variable(torch.from_numpy(t_tensor).float(), requires_grad=True).to(device)
        t0_tensor = Variable(torch.from_numpy(t0_tensor).float(), requires_grad=True).to(device)
        t1_tensor = Variable(torch.from_numpy(t1_tensor).float(), requires_grad=True).to(device)
    
        for batch in range(0,num_batches):
        
            # Extract random points
            indices = random.sample(index_list,k=batch_size)
            x_batch = x_tensor[indices]; y_batch = y_tensor[indices]
            t_batch = t_tensor[indices]; t0_batch = t0_tensor[indices]; t1_batch = t1_tensor[indices]
            pt_all_zeros_batch = pt_all_zeros[indices]
            u0_batch = u0[indices,:]; u1_batch = u1[indices,:]
            X_vector_batch = X_vector[indices]; Y_vector_batch = Y_vector[indices]
            u0_vector_batch = u0_vector[indices]; u1_vector_batch = u1_vector[indices]
        
        

            # Compute DeepONet solutions at t=0 and t=1
            u_branch0 = Cty_branch0(u0_batch)
            u_branch1 = Cty_branch1(u1_batch)    
            u_trunk0 = Cty_trunk(X_vector_batch,Y_vector_batch,t0_batch) 
            u_trunk1 = Cty_trunk(X_vector_batch,Y_vector_batch,t1_batch)    

            bc_out0 = torch.sum( (u_branch0*u_branch1) * u_trunk0, dim=-1).unsqueeze(1)
            bc_out1 = torch.sum( (u_branch0*u_branch1) * u_trunk1, dim=-1).unsqueeze(1)
    
            # Evaluate boundary conditions
            mse_u0 = mse_cost_function(bc_out0, u0_vector_batch)
            mse_u1 = mse_cost_function(bc_out1, u1_vector_batch)
        
        
            # Evalate physics-informed loss
            out_0 = HJ(x_batch, y_batch, t_batch, u0_batch, u1_batch, HJ_branch0, HJ_branch1, HJ_trunk) 
            out_1 = Cty(x_batch, y_batch, t_batch, u0_batch, u1_batch, HJ_branch0, HJ_branch1, HJ_trunk, Cty_branch0, Cty_branch1, Cty_trunk)
            mse_HJ = mse_cost_function(out_0, pt_all_zeros_batch)
            mse_Cty = mse_cost_function(out_1, pt_all_zeros_batch)
        
        
            # Evaluate total loss
            loss =   (coefficients[0])*mse_HJ  +  (coefficients[1])*mse_Cty  +  (coefficients[2])*mse_u0  +  (coefficients[3])*mse_u1 


            optimizer_HJ_branch0.zero_grad(); optimizer_HJ_branch1.zero_grad(); optimizer_HJ_trunk.zero_grad()
            optimizer_Cty_branch0.zero_grad(); optimizer_Cty_branch1.zero_grad(); optimizer_Cty_trunk.zero_grad() 
        
            loss.backward() 
            
            # Clip gradient
            torch.nn.utils.clip_grad_norm_(HJ_branch0.parameters(), 1); torch.nn.utils.clip_grad_norm_(HJ_branch1.parameters(), 1)
            torch.nn.utils.clip_grad_norm_(HJ_trunk.parameters(), 1);torch.nn.utils.clip_grad_norm_(Cty_branch0.parameters(), 1)
            torch.nn.utils.clip_grad_norm_(Cty_branch1.parameters(), 1); torch.nn.utils.clip_grad_norm_(Cty_trunk.parameters(), 1)

            # Take optimizer step
            optimizer_HJ_branch0.step(); optimizer_HJ_branch1.step(); optimizer_HJ_trunk.step()
            optimizer_Cty_branch0.step(); optimizer_Cty_branch1.step(); optimizer_Cty_trunk.step() 

            
        # Add most recent loss values to loss_matrix for changing the coefficients
        if batch % 200 == 0: 
            loss_matrix = np.roll(loss_matrix,-1, axis=0)
            loss_matrix[num_rows-1,0] = mse_HJ; loss_matrix[num_rows-1,1] = mse_Cty
            loss_matrix[num_rows-1,2] = mse_u0; loss_matrix[num_rows-1,3] = mse_u1
        
        with torch.autograd.no_grad():
            print(epoch,"Training Loss:", '{:.4e}'.format(loss.data) )
            
         
    