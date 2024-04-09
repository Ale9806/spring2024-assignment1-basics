import torch.nn as nn 
from src.transformer.activation_function import GeLu
  
class FFN(nn.Module): 
    def __init__(self,w1,w2): 
        super(FFN, self).__init__() 
        self.GeLu = GeLu()
        self.W1   = w1
        self.W2   = w2
  
    def forward(self, x):
        return  self.GeLu(x@self.W1.T)@self.W2.T