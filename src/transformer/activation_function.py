import torch 
import torch.nn as nn 
  
class GeLu(nn.Module): 
    def __init__(self): 
        super(GeLu, self).__init__() 
  
    def forward(self, x): 
        sqrt_2 = torch.sqrt(torch.tensor(2))
        return x*0.5* (1+torch.erf(x/sqrt_2))
    

class Softmax(nn.Module): 
    def __init__(self): 
        super(Softmax, self).__init__() 
  
    def forward(self, x): 
        x_max = torch.max(x)
        exp_x = torch.exp(x - x_max)
        return exp_x / torch.sum(exp_x)
