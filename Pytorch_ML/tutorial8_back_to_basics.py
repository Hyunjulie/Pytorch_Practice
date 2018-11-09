# From Morvan Zhou's PyTorch Tutorial github 
# Practing & adding some reusable lines of code 

import torch 
import matplotlib.pyplot as plt 

torch.manual_seed(999)

#Making up data 
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1) # Tensor, shape(100, 1)
y = x.pow(2) + 0.2*torch.rand(x.size9)

