from torch import nn
import networks

class CWVAE(nn.Module):
    
    def __init__(self, config):
        super(CWVAE, self).__init__()
        self.__use_amp = True if config.precision==16 else False
        self._config = config
        self.encoder = networks.ConvEncoder(config.grayscale, 
                                            config.cnn_depth,
                                            config.act,
                                            config.encoder_kernels)
        
        