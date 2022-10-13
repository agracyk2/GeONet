import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np;
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# Declare trunk neural network architecture
class Net_trunk(nn.Module):
    def __init__(self):
        super(Net_trunk, self).__init__()
        self.hidden_layer1 = nn.Linear(3,120, bias=True)
        self.hidden_layer2 = nn.Linear(120,120, bias=True); self.hidden_layer3 = nn.Linear(120,120, bias=True)
        self.hidden_layer4 = nn.Linear(120,120, bias=True); self.hidden_layer5 = nn.Linear(120,120, bias=True)
        self.hidden_layer6 = nn.Linear(120,120, bias=True); self.hidden_layer7 = nn.Linear(120,120, bias=True)
        self.output_layer = nn.Linear(120,120, bias=True)

    def forward(self, x,y,t):
        inputs = torch.cat([x,y,t],1) # combined three arrays of 1 columns each to one array of 3 columns
        layer1_out = torch.tanh(self.hidden_layer1(inputs)); layer2_out = torch.tanh(self.hidden_layer2(layer1_out))
        layer3_out = torch.tanh(self.hidden_layer3(layer2_out)); layer4_out = torch.tanh(self.hidden_layer4(layer3_out))
        layer5_out = torch.tanh(self.hidden_layer5(layer4_out)); layer6_out = torch.tanh(self.hidden_layer6(layer5_out))
        layer7_out = torch.tanh(self.hidden_layer7(layer6_out))
        output = self.output_layer(layer7_out)
        return output

    # Initialize weights
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()
                
# Declare branch neural network architecture
class Net_branch(nn.Module):
    def __init__(self):
        super(Net_branch, self).__init__()
        # 900 corresponds to vectorized branch input of Gaussian mixture
        self.hidden_layer1 = nn.Linear(900,180, bias=True); self.hidden_layer2 = nn.Linear(180,180, bias=True)
        self.hidden_layer3 = nn.Linear(180,180, bias=True); self.hidden_layer4 = nn.Linear(180,180, bias=True)
        self.hidden_layer5 = nn.Linear(180,180, bias=True); self.output_layer = nn.Linear(180,120, bias=True)

    def forward(self, x):
        inputs = torch.cat([x],1) 
        layer1_out = torch.tanh(self.hidden_layer1(inputs)); layer2_out = torch.tanh(self.hidden_layer2(layer1_out))
        layer3_out = torch.tanh(self.hidden_layer3(layer2_out)); layer4_out = torch.tanh(self.hidden_layer4(layer3_out))
        layer5_out = torch.tanh(self.hidden_layer5(layer4_out))
        output = self.output_layer(layer5_out)
        return output

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()
                
# Create neural neyworks

# Hamilton-Jacobi neural networks
HJ_branch0 = Net_branch()
HJ_branch1 = Net_branch()
HJ_branch0.to(device)
HJ_branch1.to(device)
HJ_trunk = Net_trunk()
HJ_trunk.to(device)

# Continuity neural networks
Cty_branch0 = Net_branch()
Cty_branch1 = Net_branch()
Cty_branch0.to(device)
Cty_branch1.to(device)
Cty_trunk = Net_trunk()
Cty_trunk.to(device)


mse_cost_function = torch.nn.MSELoss() # Mean squared error
lrate = 0.0001
optimizer_HJ_branch0 = torch.optim.Adam(HJ_branch0.parameters(), lr=lrate)
optimizer_HJ_branch1 = torch.optim.Adam(HJ_branch1.parameters(), lr=lrate)
optimizer_HJ_trunk = torch.optim.Adam(HJ_trunk.parameters(), lr=lrate)
optimizer_Cty_branch0 = torch.optim.Adam(Cty_branch0.parameters(), lr=lrate)
optimizer_Cty_branch1 = torch.optim.Adam(Cty_branch1.parameters(), lr=lrate)
optimizer_Cty_trunk = torch.optim.Adam(Cty_trunk.parameters(), lr=lrate)
