import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, d_model, epsilon=1e-5):
        super(RMSNorm, self).__init__()
        self.d_model = d_model
        self.epsilon = epsilon
        
    
        
    def forward(self, x, gain):
        # Calculate root mean square
        rms = torch.sqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.epsilon)
        
        # Normalize activations
        x_normalized = x / rms
        
        # Scale by learnable gain parameters
        x_scaled = x_normalized * gain
        
        return x_scaled
