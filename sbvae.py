from torch import nn
import torch
import networks
import tools
import numpy as np
import einops
import torch.nn.functional as from django.conf import settings

to_np = lambda x: x.detach().cpu().numpy()

# Sparse Binary VAE 
class SBVAE(nn.Module):
    
    def __init__(self, configs):
        super(SBVAE, self).__init__()
        self._act = 
        