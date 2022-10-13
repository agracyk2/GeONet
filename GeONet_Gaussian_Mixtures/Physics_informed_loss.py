import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Hamilton-Jacobi physics-informed loss
def HJ(x, y, t, u0, u1, HJ_branch0, HJ_branch1, HJ_trunk):
    u_branch0 = HJ_branch0(u0)
    u_branch1 = HJ_branch1(u1)
    u_trunk = HJ_trunk(x,y,t)

    # Compute DeepONet solution to HJ equation
    u = torch.sum( u_branch0*u_branch1 * u_trunk, dim=-1).unsqueeze(1)
      
    # Automatic differentiation
    u_t = torch.autograd.grad(u_trunk.sum(), t, create_graph=True)[0]
    u_x = torch.autograd.grad(u_trunk.sum(), x, create_graph=True)[0]
    u_y = torch.autograd.grad(u_trunk.sum(), y, create_graph=True)[0]
    
    # DeepONet computation
    U_t = torch.sum( (u_branch0*u_branch1) * u_t, dim=-1).unsqueeze(1)
    U_x = torch.sum( (u_branch0*u_branch1) * u_x, dim=-1).unsqueeze(1)
    U_y = torch.sum( (u_branch0*u_branch1) * u_y, dim=-1).unsqueeze(1)

    # Return PDE residual
    pde = U_t + (1/2)*(U_x**2 + U_y**2)
    return pde


# Continuity physics-informed loss
def Cty(x, y, t, u0, u1, HJ_branch0, HJ_branch1, HJ_trunk, Cty_branch0,  Cty_branch1, Cty_trunk):
    u_branch0 = HJ_branch0(u0)  # u is HJ solution
    u_branch1 = HJ_branch1(u1)
    u_trunk = HJ_trunk(x,y,t)  
    v_branch0 = Cty_branch0(u0)  # v is continuity solution
    v_branch1 = Cty_branch1(u1)
    v_trunk = Cty_trunk(x,y,t)  

    # Compute DeepONet solution to continuity equation
    v = torch.sum( v_branch0*v_branch1 * v_trunk, dim=-1).unsqueeze(1)

    # Automatic differentiation
    u_x = torch.autograd.grad(u_trunk.sum(), x, create_graph=True)[0]
    u_y = torch.autograd.grad(u_trunk.sum(), y, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]
    
    v_t = torch.autograd.grad(v_trunk.sum(), t, create_graph=True)[0]
    v_x = torch.autograd.grad(v_trunk.sum(), x, create_graph=True)[0]
    v_y = torch.autograd.grad(v_trunk.sum(), y, create_graph=True)[0]
    
    # Construct DeepONet solutions with evaluated derivatives
    
    # Continuity DeepONets
    V_t = torch.sum( (v_branch0*v_branch1) * v_t, dim=-1).unsqueeze(1)
    V_x = torch.sum( (v_branch0*v_branch1) * v_x, dim=-1).unsqueeze(1)
    V_y = torch.sum( (v_branch0*v_branch1) * v_y, dim=-1).unsqueeze(1)  
    
    # HJ DeepONets
    U_x = torch.sum( (u_branch0*u_branch1) * u_x, dim=-1).unsqueeze(1)
    U_y = torch.sum( (u_branch0*u_branch1) * u_y, dim=-1).unsqueeze(1)
    U_xx = torch.sum( (u_branch0*u_branch1) * u_xx, dim=-1).unsqueeze(1)
    U_yy = torch.sum( (u_branch0*u_branch1) * u_yy, dim=-1).unsqueeze(1)

    # Compute divergence with product rule
    div_x = V_x*U_x + v*U_xx
    div_y = V_y*U_y + v*U_yy

    # Return PDE residual
    pde = V_t + div_x + div_y
    return pde