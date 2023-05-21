import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torch.utils.data as data
import lightning.pytorch as pl 

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(28*28, 64), nn.ReLU(), nn.Linear(64,3))
        
    def forward(self,x):
        return self.l1(x)
    
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(3,64), nn.ReLU(), nn.Linear(64,28*28))
        
    def forward(self, x):
        return self.l1(x)
    
class LitAutoEncoder(pl.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        val_loss = F.mse_loss(x_hat, x)
        self.log("val_loss", val_loss)
    
    def test_step(self, batch, batch_idx):
        x, y = batchx = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        test_loss = F.mse_loss(x_hat, x)
        self.log("test_loss", test_loss)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer 
    
transform = transforms.ToTensor()


train_set = MNIST(root='MNIST', download=True, train=True, transform=transform)
test_set = MNIST(root='MNIST', download=True, train=False, transform=transform)

train_set_size = int(len(train_set) * 0.8)
valid_set_size = len(train_set) - train_set_size

seed = torch.Generator().manual_seed(42)
train_set, valid_set = data.random_split(train_set, [train_set_size, valid_set_size], generator=seed)
autoencoder = LitAutoEncoder(Encoder(), Decoder())

trainer = pl.Trainer()
trainer.fit(model=autoencoder, train_dataloaders=DataLoader(train_set), val_dataloaders=DataLoader(valid_set))

trainer.test(autoencoder, dataloaders=DataLoader(test_set))

