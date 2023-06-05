import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions as torchd

import tools
import einops
from typing import Union
from einops.layers.torch import Rearrange, Reduce

Activation = Union[str, nn.Module]

_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
    'elu': nn.ELU()
}

class RSSM(nn.Module):

  def __init__(
      self, stoch=30, deter=200, hidden=200, layers_input=1, layers_output=1,
      discrete=False, act=nn.ELU,
      mean_act='none', std_act='softplus', min_std=0.1,
      cell='gru', num_actions=None, embed=None, device=None):
    super(RSSM, self).__init__()
    self._stoch = stoch
    self._deter = deter
    self._hidden = hidden
    self._min_std = min_std
    self._layers_input = layers_input
    self._layers_output = layers_output
    self._discrete = discrete
    self._act = act
    self._mean_act = mean_act
    self._std_act = std_act
    self._embed = embed
    self._device = device

    inp_layers = []

    if self._discrete:
      stoch_dim = self._stoch * self._discrete
    else:
      stoch_dim = self._stoch
    self.state_size = self._deter + stoch_dim
    
    inp_dim = stoch_dim + self.state_size + num_actions

    for i in range(self._layers_input):
      inp_layers.append(nn.Linear(inp_dim, self._hidden))
      inp_layers.append(self._act())
      if i == 0:
        inp_dim = self._hidden
    self._inp_layers = nn.Sequential(*inp_layers)

    if cell == 'gru':
      self._cell = GRUCell(self._hidden, self._deter)
    elif cell == 'gru_layer_norm':
      self._cell = GRUCell(self._hidden, self._deter, norm=True)
    else:
      raise NotImplementedError(cell)

    img_out_layers = []
    inp_dim = self._deter
    for i in range(self._layers_output):
      img_out_layers.append(nn.Linear(inp_dim, self._hidden))
      img_out_layers.append(self._act())
      if i == 0:
        inp_dim = self._hidden
    self._img_out_layers = nn.Sequential(*img_out_layers)

    obs_out_layers = []
    inp_dim = self._deter + self._embed
    for i in range(self._layers_output):
      obs_out_layers.append(nn.Linear(inp_dim, self._hidden))
      obs_out_layers.append(self._act())
      if i == 0:
        inp_dim = self._hidden
    self._obs_out_layers = nn.Sequential(*obs_out_layers)

    if self._discrete:
      self._ims_stat_layer = nn.Linear(self._hidden, self._stoch*self._discrete)
      self._obs_stat_layer = nn.Linear(self._hidden, self._stoch*self._discrete)
    else:
      self._ims_stat_layer = nn.Linear(self._hidden, 2*self._stoch)
      self._obs_stat_layer = nn.Linear(self._hidden, 2*self._stoch)
 
  def initial(self, batch_size):
    deter = torch.zeros(batch_size, self._deter).to(self._device)
    if self._discrete:
      state = dict(
          logit=torch.zeros([batch_size, self._stoch, self._discrete]).to(self._device),
          stoch=torch.zeros([batch_size, self._stoch, self._discrete]).to(self._device),
          deter=deter)
    else:
      state = dict(
          mean=torch.zeros([batch_size, self._stoch]).to(self._device),
          std=torch.zeros([batch_size, self._stoch]).to(self._device),
          stoch=torch.zeros([batch_size, self._stoch]).to(self._device),
          deter=deter)
    return state

  def initial_context(self, batch_size, seq_len):
    context = torch.zeros(batch_size, seq_len, self.state_size).to(self._device)
    return context

  def observe(self, embed, context, action, state=None):
    B, T, F = embed.shape
    swap = lambda x: x.permute([1,0] + list(range(2, len(x.shape))))

    if state is None:
      state = self.initial(B)
    if context is None:
      context = self.initial_context(B, T)
    else:
      # Context was repeated in the above layer and does not match timestep for this layer
      # Trim context to match the embedding 
      context = context[:, :T, :].clone().detach().requires_grad_(False)
    embed, action, context = swap(embed), swap(action), swap(context)
    post, prior = tools.static_scan(
        lambda prev_state, context, prev_act, embed: self.obs_step(
            prev_state[0], context, prev_act, embed),
        (context, action, embed), (state, state))
    post = {k: swap(v) for k, v in post.items()}
    prior = {k: swap(v) for k, v in prior.items()}
    return post, prior

  def imagine(self, context, action, state=None):
    B, T, F = action.shape
    swap = lambda x: x.permute([1,0] + list(range(2, len(x.shape))))

    if state is None:
      state = self.initial(B)
    if context is None:
      context = self.initial_context(B, T)
    else:
      # Context was repeated in the above layer and does not match timestep for this layer
      # Trim context to match the embedding 
      context = context[:, :T, :] 
    assert isinstance(state, dict), state
    action, context = swap(action), swap(context)
    prior = tools.static_scan(
      lambda prev_state, context, prev_act: self.img_step(
        prev_state, context, prev_act),
      (context, action),  state)
    prior = prior[0]
    prior = {k: swap(v) for k, v in prior.items()}
    return prior

  def get_feat(self, state):
    stoch = state['stoch']
    if self._discrete:
      shape = list(stoch.shape[:-2]) + [self._stoch * self._discrete]
      stoch = stoch.reshape(shape)
    return torch.cat([stoch, state['deter']], -1)

  def get_dist(self, state, dtype=None):
    if self._discrete:
      logit = state['logit']
      dist = torchd.independent.Independent(tools.OneHotDist(logit), 1)
    else:
      mean, std = state['mean'], state['std']
      dist = tools.ContDist(torchd.independent.Independent(
          torchd.normal.Normal(mean, std), 1))
    return dist

  def obs_step(self, prev_state, context, prev_action, embed, sample=True):
    prior = self.img_step(prev_state, context, prev_action, None, sample)
    x = torch.cat([prior['deter'], embed], -1)
    x = self._obs_out_layers(x)
    stats = self._suff_stats_layer('obs', x)
    if sample:
      stoch = self.get_dist(stats).sample()
    else:
      stoch = self.get_dist(stats).mode()
    post = {'stoch': stoch, 'deter': prior['deter'], **stats}
    return post, prior

  def img_step(self, prev_state, context, prev_action, embed=None, sample=True):
    prev_stoch = prev_state['stoch']
    if self._discrete:
      prev_stoch = einops.rearrange(prev_stoch, 'b d f -> b (d f)')

    x = torch.cat([prev_stoch, context, prev_action], -1)
    x = self._inp_layers(x)
    deter = prev_state['deter']
    x, deter = self._cell(x, [deter])
    deter = deter[0]  # Keras wraps the state in a list.
    x = self._img_out_layers(x)
    stats = self._suff_stats_layer('ims', x)
    if sample:
      stoch = self.get_dist(stats).sample()
    else:
      stoch = self.get_dist(stats).mode()
    prior = {'stoch': stoch, 'deter': deter, **stats}
    return prior

  def _suff_stats_layer(self, name, x):
    if self._discrete:
      if name == 'ims':
        x = self._ims_stat_layer(x)
      elif name == 'obs':
        x = self._obs_stat_layer(x)
      else:
        raise NotImplementedError
      logit = x.reshape(list(x.shape[:-1]) + [self._stoch, self._discrete])
      return {'logit': logit}
    else:
      if name == 'ims':
        x = self._ims_stat_layer(x)
      elif name == 'obs':
        x = self._obs_stat_layer(x)
      else:
        raise NotImplementedError
      mean, std = torch.split(x, [self._stoch]*2, -1)
      mean = {
          'none': lambda: mean,
          'tanh5': lambda: 5.0 * torch.tanh(mean / 5.0),
      }[self._mean_act]()
      std = {
          'softplus': lambda: F.softplus(std),
          'abs': lambda: torch.abs(std + 1),
          'sigmoid': lambda: torch.sigmoid(std),
          'sigmoid2': lambda: 2 * torch.sigmoid(std / 2),
      }[self._std_act]()
      std = std + self._min_std
      return {'mean': mean, 'std': std}

  def kl_loss(self, post, prior, forward, balance, free, scale, agg='mean'):
    kld = torchd.kl.kl_divergence
    dist = lambda x: self.get_dist(x)
    sg = lambda x: {k: v.detach() for k, v in x.items()}
    lhs, rhs = (prior, post) if forward else (post, prior)
    mix = balance if forward else (1 - balance)
    if balance == 0.5:
      value = kld(dist(lhs) if self._discrete else dist(lhs)._dist,
                  dist(rhs) if self._discrete else dist(rhs)._dist)
      value = torch.mean( torch.sum(value, dim=1))
      loss = torch.maximum(value, free)
      # loss = torch.mean(value)
      
    else:
      value_lhs = value = kld(dist(lhs) if self._discrete else dist(lhs)._dist,
                              dist(sg(rhs)) if self._discrete else dist(sg(rhs))._dist)
      value_rhs = kld(dist(sg(lhs)) if self._discrete else dist(sg(lhs))._dist,
                      dist(rhs) if self._discrete else dist(rhs)._dist)
      if agg == 'mean':
        loss_lhs = torch.maximum(torch.mean(value_lhs), torch.Tensor([free])[0])
        loss_rhs = torch.maximum(torch.mean(value_rhs), torch.Tensor([free])[0])
      elif agg == 'sum':
        loss_lhs = torch.maximum(value_lhs, torch.Tensor([free]).to(self._device)).sum()
        loss_rhs = torch.maximum(value_rhs, torch.Tensor([free]).to(self._device)).sum()
      loss = mix * loss_lhs + (1 - mix) * loss_rhs
    loss *= scale
    return loss, value


"""
    Multi-level Video Encoder.
    1. Extracts hierarchical features from a sequence of observations.
    2. Encodes observations using Conv layers, uses them directly for the bottom-most level.
    3. Uses dense features for each level of the hierarchy above the bottom-most level.
"""
  
class HierarchicalEncoder(nn.Module):

  def __init__(self, levels, tmp_abs_factor, dense_layers=3,
    hidden_size=100, channels_mult=1, channels=3, input_size=(64, 64), depth=32,
    act=nn.ReLU, kernels=(4, 4, 4, 4)):
    """
      Arguments:
          levels : int
              Number of levels in the hierarchy
          tmp_abs_factor : int
              Temporal abstraction factor used at each level
          dense_layers : int
              Number of dense hidden layers at each level
          hidden_size : int
              Size of dense hidden embeddings
          channels_mult: int
              Multiplier for the number of channels in the conv encoder
    """
    super(HierarchicalEncoder, self).__init__()
    self._levels = levels
    self._tmp_abs_factor = tmp_abs_factor
    self._dense_layers = dense_layers
    self._channels_mult = channels_mult
    self._activation = act
    self._kwargs = dict(strides=2, activation=self._activation, use_bias=True)
 
    assert levels >= 1, "levels should be >=1, found {}".format(levels)
    assert tmp_abs_factor >= 1, "tmp_abs_factor should be >=1, found {}".format(
        tmp_abs_factor
    )
    assert (
        not dense_layers or hidden_size
    ), "embed_size={} invalid for Dense layer".format(embed_size)

    self.conv_encoder = ConvEncoder(channels=channels,
                               depth=depth,
                               act=getattr(nn,act),
                               kernels=kernels)
    shape = (*input_size, channels)
    # Get the output embedding size of ConvEncoder by testing it
    testObs = torch.rand(1, 1, *shape) 
    embed_size = self.conv_encoder(testObs).shape[-1]
    self.fc_encoders = nn.ModuleList()
    self.embed_size = embed_size
    
    for i in range(1, levels):
      encoder = build_mlp(embed_size,
                          embed_size, 
                          dense_layers,
                          hidden_size,
                          act)
      self.fc_encoders.append(encoder)
  
  def __call__(self, obs):
    """
        Arguments:
            obs : Tensor
                Un-flattened observations (videos) of shape (batch size, timesteps, width, height, channel)
    """
    outputs = []
    embedding = self.conv_encoder(obs)
    outputs.append(embedding)
    
    for level in range(self._levels-1):
      embedding = self.fc_encoders[level](embedding.detach())

      # embedding is [B, T, F] dimension
      # To reduce with _tmp_abs_factor^level, we need to pad T dimension of embedding
      # such that it is divisible with timesteps_to_merge
      timesteps_to_merge = np.power(self._tmp_abs_factor, level+1)
      timesteps_to_pad = np.mod(
        timesteps_to_merge - np.mod(embedding.shape[1], timesteps_to_merge),
        timesteps_to_merge
      )
      pad = (0, 0, 0, timesteps_to_pad, 0, 0)
      embedding = F.pad(embedding, pad, "constant", 0)
      embedding = einops.reduce(embedding, 'b (t t2) f -> b t f', 'sum', t2=timesteps_to_merge)
      outputs.append(embedding)
    return outputs

class ConvEncoder(nn.Module):

  def __init__(self, emb_dim, channels=3,
               depth=32, act=nn.ReLU, kernels=(4, 4, 4, 4)):
    super(ConvEncoder, self).__init__()
    self._act = act
    self._depth = depth
    self._kernels = kernels

    layers = []
    for i, kernel in enumerate(self._kernels):
      if i == 0:
        inp_dim = channels
      else:
        inp_dim = 2 ** (i-1) * self._depth
      depth = 2 ** i * self._depth
      layers.append(nn.Conv2d(inp_dim, depth, kernel, 2))
      layers.append(act())
    layers.append(Rearrange('b c h w -> b (c h w)'))  
    
    layers.append(nn.LazyLinear (emb_dim))
    self.layers = nn.Sequential(*layers)

  def __call__(self, obs):
    x = einops.rearrange(obs, 'b h w c -> b  c h w')
    x = self.layers(x)
    # x = einops.rearrange(x, 'b c h w -> b (c h w)', b=obs.shape[0])
    return x  

class preprocessAE(nn.Module):
  
  def encode(self, obs):
    # obs = obs.clone()
    obs = obs  - 0.5 
    obs = obs * 2.0
    return obs

  def decode(self, obs):
    obs = obs / 2.0 + 0.5
    return obs

  def forward(self, obs):
    z = self.encode(obs)
    return self.decode(z), None,z
 
class Conv3dAE(nn.Module):
  
  def __init__(self, channels_factor=4, 
               num_conv_layers=2, 
               act=nn.ELU,
               kernels=(3,3,3),
               stride=(2,2,2),
               input_width=64,
               input_height=64,
               input_channels=1,
               temp_abs_factor=4):
    super(Conv3dAE, self).__init__()
    
    c_hid = channels_factor * input_channels 
    
    self.encoder = nn.Sequential(
      # nn.BatchNorm3d(input_channels, affine=False), 
        nn.Conv3d(input_channels, c_hid, kernel_size=3, padding=1, stride=2),  # 64x64 => 32x32
        act(),
        nn.Conv3d(c_hid, c_hid, kernel_size=3, padding=1),
        act(),
        nn.Conv3d(c_hid, channels_factor * c_hid, kernel_size=3, padding=1, stride=2),  # 32x32 => 16x16
        act(),
        nn.Conv3d(channels_factor * c_hid, channels_factor * c_hid, kernel_size=3, padding=1),
        # nn.BatchNorm3d(channels_factor*c_hid, affine=False)
        nn.Tanh()
        )


    self.decoder = nn.Sequential(
      # nn.BatchNorm3d(channels_factor*c_hid, affine=False),
        nn.ConvTranspose3d(
            channels_factor * c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2
        ),  # 16x16 => 32x32
        act(),
        nn.Conv3d( c_hid,  c_hid, kernel_size=3, padding=1),
        act(),
        nn.ConvTranspose3d(c_hid, input_channels, kernel_size=3, output_padding=1, padding=1, stride=2),  # 32x32 => 64x64
        act(),
        nn.Conv3d(input_channels, input_channels, kernel_size=3, padding=1),
        # nn.BatchNorm3d(input_channels, affine=False)
    )
    
  def forward(self, x):
    # Assume x is (b t h w c)
    x = einops.rearrange(x, 'b t h w c -> b c t h w' ) 
    z = self.encoder(x)
    recon = self.decoder(z)
    recon = einops.rearrange(recon, 'b c t h w -> b t h w c')
    z = einops.rearrange(z, 'b c t h w -> b t h w c ')
    return recon, z
  
  def decode(self, emb):
    # Assume emb is (b t h w c)
    z = einops.rearrange(emb, 'b t h w c -> b c t h w')
    recon = self.decoder(z)
    recon = einops.rearrange(recon, 'b c t h w -> b t h w c')
    return recon 
    
  def encode(self, x):
    # Assume x is (b t h w c)
    x = einops.rearrange(x, 'b t h w c -> b c t h w')
    z = self.encoder(x)
    z = einops.rearrange(z, 'b c t h w -> b t h w c ')
    return z



class Conv3dVAE(nn.Module):
  
  def __init__(self, channels_factor=4, 
               num_conv_layers=2, 
               act=nn.ELU,
               kernels=(3,3,3),
               stride=(2,2,2),
               input_width=64,
               input_height=64,
               input_channels=1,
               emb_shape=(64,64,1),
               discrete=0,
               hid_factor=1,
               temp_abs_factor=4):
    super(Conv3dVAE, self).__init__()
    
    
    self.discrete = discrete
    
    if discrete:
      discrete_factor = discrete
    else:
      discrete_factor = 1
      
    c_hid = channels_factor * input_channels * hid_factor
      
    if self.discrete:
      self.encoder = nn.Sequential(
        nn.Conv3d(input_channels, c_hid, kernel_size=3, padding=1, stride=2),  # 64x64 => 32x32
        act(),
        nn.Conv3d(c_hid, c_hid, kernel_size=3, padding=1),
        act(),
        nn.Conv3d(c_hid, channels_factor * c_hid, kernel_size=3, padding=1, stride=2),  # 32x32 => 16x16
        act(),
        nn.Conv3d( channels_factor * c_hid, channels_factor * c_hid, kernel_size=3, padding=1),
        Rearrange('b (c1 c2) t h w ->  b t h w c1 c2', c2=discrete_factor),
        ) 
    else:
      self.encoder = nn.Sequential(
          nn.Conv3d(input_channels, c_hid, kernel_size=3, padding=1, stride=2),  # 64x64 => 32x32
          act(),
          nn.Conv3d(c_hid, c_hid, kernel_size=3, padding=1),
          act(),
          nn.Conv3d(c_hid, channels_factor * c_hid, kernel_size=3, padding=1, stride=2),  # 32x32 => 16x16
          act(),
          nn.Conv3d(channels_factor * c_hid, 2*channels_factor * c_hid, kernel_size=3, padding=1),
          Rearrange('b (c a) t h w -> a b t h w c', a=2),
          )


    self.decoder = nn.Sequential(
      # nn.BatchNorm3d(channels_factor*c_hid, affine=False),
        nn.ConvTranspose3d(
            channels_factor * c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2
        ),  # 16x16 => 32x32
        act(),
        nn.Conv3d( c_hid,  c_hid, kernel_size=3, padding=1),
        act(),
        nn.ConvTranspose3d(c_hid, input_channels, kernel_size=3, output_padding=1, padding=1, stride=2),  # 32x32 => 64x64
        act(),
        nn.Conv3d(input_channels, input_channels, kernel_size=3, padding=1),
        # nn.BatchNorm3d(input_channels, affine=False)
    )

    
  def forward(self, x):
    dist = self.encode_dist(x)
    z = dist.rsample()
    if self.discrete:
      z = einops.rearrange(z, 'b t h w c1 c2 -> b t h w (c1 c2)')
    recon = self.decode(z)
    return recon, dist, z
  
  def decode(self, emb):
    # Assume emb is (b t h w c)
    z = einops.rearrange(emb, 'b t h w c -> b c t h w')
    recon = self.decoder(z)
    recon = einops.rearrange(recon, 'b c t h w -> b t h w c')
    return recon 
    
  def encode_dist(self, x):
    # Assume x is (b t h w c)
    x = einops.rearrange(x, 'b t h w c -> b c t h w')
    z = self.encoder(x)
    if self.discrete:
      dist = torchd.independent.Independent(
        torchd.OneHotCategoricalStraightThrough(logits=z),3)
    else:
      mu, logvar = z[0], z[1]
      std = torch.exp(0.5*logvar)
      dist = tools.ContDist(torchd.independent.Independent(
      torchd.normal.Normal(mu, std), 2)) 
    return dist
    
  def kl_divergence(self, dist):
    if self.discrete:
      logits = torch.ones_like(dist.mean)
      prior = torchd.independent.Independent(
        torchd.OneHotCategoricalStraightThrough(logits=logits),3) 
      kld = torchd.kl_divergence(dist, prior)
    else:
      sigma = dist._dist.stddev
      sigma = torch.log(sigma) * 2
      mu = dist._dist.mean
      kld = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
    return kld
  
  def encode(self, x):
    dist = self.encode_dist(x)
    z = dist.rsample()
    if self.discrete:
      z = einops.rearrange(z, 'b t h w c1 c2 -> b t h w (c1 c2)')
    return z 
  
class LocalConvEncoder(nn.Module):
  
  def __init__(self, 
               output_dim,
               channels_factor=4, 
               act=nn.GELU,
               kernels=(3,3,3),
               stride=(2,2,2),
               input_width=64,
               input_height=64,
               input_channels=1,
               temp_abs_factor=4):
    super(LocalConvEncoder, self).__init__()
    

    c_hid = channels_factor * input_channels 
    cnn_output_dim = input_width * input_height * input_channels //4
    
    self.encoder = nn.Sequential(
        nn.Conv2d(input_channels, c_hid, kernel_size=3, padding=1, stride=2),  # 64x64 => 32x32
        act(),
        nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
        act(),
        nn.Conv2d(c_hid, channels_factor * c_hid, kernel_size=3, padding=1, stride=2),  # 32x32 => 16x16
        act(),
        nn.Conv2d(channels_factor * c_hid, channels_factor * c_hid, kernel_size=3, padding=1),
        act(),
        Rearrange('b c h w -> b (c h w)'),
        nn.Linear(cnn_output_dim, output_dim )
        )
    
  def forward(self, x):
    # Assume x is (b t h w c)
    x = einops.rearrange(x, 'b h w c -> b c h w')
    z = self.encoder(x)
    # z = einops.rearrange(z, 'b c h w -> b h w c ')
    return z
  
class LocalConvDecoder(nn.Module):
  
  def __init__(self, shape=(16, 16, 128), feat_size=232, act=nn.GELU):
    super(LocalConvDecoder, self).__init__()
    self._shape = shape
    self.height, self.width, self.channels = shape
    cnnt_layers = []
    self.linear_layer = nn.Linear(feat_size, self.channels  * self.width * self.height//4)
    self.cnnt_layers = nn.Sequential(
        nn.ConvTranspose2d(
            self.channels * 4, self.channels *2, kernel_size=3, output_padding=1, padding=1, stride=2
        ),  # 16x16 => 32x32
        act(),
        nn.Conv2d( self.channels*2,  self.channels*2, kernel_size=3, padding=1),
        act(),
        nn.ConvTranspose2d(self.channels*2, self.channels, kernel_size=3, output_padding=1, padding=1, stride=2),  # 32x32 => 64x64
        act(),
        nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1),
        # nn.Tanh(),  # The input images is scaled between -1 and 1, hence the output has to be bounded as well
    )
  
  def __call__(self, features):
    x = self.linear_layer(features)
    x = einops.rearrange(x, 'b t (h w c) -> (b t) c h w', c=self.channels*4, h=self.height//4, w=self.width//4)
    x = self.cnnt_layers(x)
    mean = einops.rearrange(x, '(b t) c h w -> b t h w c', b = features.shape[0])
    return tools.ContDist(torchd.independent.Independent(
      torchd.normal.Normal(mean,1), len(self._shape)
    ))
      
  


class ConvDecoder(nn.Module):

  def __init__(
      self, inp_depth,
      depth=32, act=nn.GELU, shape=(3, 64, 64), kernels=(5, 5, 6, 6),
      thin=True):
    super(ConvDecoder, self).__init__()
    self._inp_depth = inp_depth
    self._act = act
    self._depth = depth
    self._shape = shape
    self._kernels = kernels
    self._thin = thin

    if self._thin:
      self._linear_layer = nn.Linear(inp_depth, 32 * self._depth)
    else:
      self._linear_layer = nn.Linear(inp_depth, 128 * self._depth)
    inp_dim = 32 * self._depth

    cnnt_layers = []
    for i, kernel in enumerate(self._kernels):
      depth = 2 ** (len(self._kernels) - i - 2) * self._depth
      act = self._act
      if i == len(self._kernels) - 1:
        #depth = self._shape[-1]
        depth = self._shape[0]
        act = None
      if i != 0:
        inp_dim = 2 ** (len(self._kernels) - (i-1) - 2) * self._depth
      cnnt_layers.append(nn.ConvTranspose2d(inp_dim, depth, kernel, 2))
      
      if i != len(self._kernels) and act is not None:
        cnnt_layers.append(act())
    cnnt_layers.append(nn.Tanh())
    self._cnnt_layers = nn.Sequential(*cnnt_layers)

  def __call__(self, features, dtype=None):
    if self._thin:
      x = self._linear_layer(features)
      x = x.reshape([-1, 1, 1, 32 * self._depth])
      x = x.permute(0,3,1,2)
    else:
      x = self._linear_layer(features)
      x = x.reshape([-1, 2, 2, 32 * self._depth])
      x = x.permute(0,3,1,2)
    x = self._cnnt_layers(x)
    mean = x.reshape(features.shape[:-1] + self._shape)
    mean = mean.permute(0, 1, 3, 4, 2)
    return tools.ContDist(torchd.independent.Independent(
      torchd.normal.Normal(mean, 1), len(self._shape)))

class DenseHead(nn.Module):

  def __init__(
      self, inp_dim,
      shape, layers, units, act=nn.ELU, dist='normal', std=1.0):
    super(DenseHead, self).__init__()
    self._shape = (shape,) if isinstance(shape, int) else shape
    if len(self._shape) == 0:
      self._shape = (1,)
    self._layers = layers
    self._units = units
    self._act = act
    self._dist = dist
    self._std = std

    mean_layers = []
    for index in range(self._layers):
      mean_layers.append(nn.Linear(inp_dim, self._units))
      mean_layers.append(act())
      if index == 0:
        inp_dim = self._units
    mean_layers.append(nn.Linear(inp_dim, np.prod(self._shape)))
    self._mean_layers = nn.Sequential(*mean_layers)

    if self._std == 'learned':
      self._std_layer = nn.Linear(self._units, np.prod(self._shape))

  def __call__(self, features, dtype=None):
    x = features
    mean = self._mean_layers(x)
    if self._std == 'learned':
      std = self._std_layer(x)
      std = torch.softplus(std) + 0.01
    else:
      std = self._std
    if self._dist == 'normal':
      return tools.ContDist(torchd.independent.Independent(
        torchd.normal.Normal(mean, std), len(self._shape)))
    if self._dist == 'huber':
      return tools.ContDist(torchd.independent.Independent(
          tools.UnnormalizedHuber(mean, std, 1.0), len(self._shape)))
    if self._dist == 'binary':
      return tools.Bernoulli(torchd.independent.Independent(
        torchd.bernoulli.Bernoulli(logits=mean), len(self._shape)))
    raise NotImplementedError(self._dist)

class ActionHead(nn.Module):

  def __init__(
      self, inp_dim, size, layers, units, act=nn.ELU, dist='trunc_normal',
      init_std=0.0, min_std=0.1, action_disc=5, temp=0.1, outscale=0):
    super(ActionHead, self).__init__()
    self._size = size
    self._layers = layers
    self._units = units
    self._dist = dist
    self._act = act
    self._min_std = min_std
    self._init_std = init_std
    self._action_disc = action_disc
    self._temp = temp() if callable(temp) else temp
    self._outscale = outscale
    pre_layers = []
    for index in range(self._layers):
      pre_layers.append(nn.Linear(inp_dim, self._units))
      pre_layers.append(act())
      if index == 0:
        inp_dim = self._units
    self._pre_layers = nn.Sequential(*pre_layers)

    if self._dist in ['tanh_normal','tanh_normal_5','normal','trunc_normal']:
      self._dist_layer = nn.Linear(self._units, 2 * self._size)
    elif self._dist in ['normal_1','onehot','onehot_gumbel']:
      self._dist_layer = nn.Linear(self._units, self._size)

  def __call__(self, features, dtype=None):
    x = features
    x = self._pre_layers(x)
    if self._dist == 'tanh_normal':
      x = self._dist_layer(x)
      mean, std = torch.split(x, 2, -1)
      mean = torch.tanh(mean)
      std = F.softplus(std + self._init_std) + self._min_std
      dist = torchd.normal.Normal(mean, std)
      dist = torchd.transformed_distribution.TransformedDistribution(
          dist, tools.TanhBijector())
      dist = torchd.independent.Independent(dist, 1)
      dist = tools.SampleDist(dist)
    elif self._dist == 'tanh_normal_5':
      x = self._dist_layer(x)
      mean, std = torch.split(x, 2, -1)
      mean = 5 * torch.tanh(mean / 5)
      std = F.softplus(std + 5) + 5
      dist = torchd.normal.Normal(mean, std)
      dist = torchd.transformed_distribution.TransformedDistribution(
          dist, tools.TanhBijector())
      dist = torchd.independent.Independent(dist, 1)
      dist = tools.SampleDist(dist)
    elif self._dist == 'normal':
      x = self._dist_layer(x)
      mean, std = torch.split(x, 2, -1)
      std = F.softplus(std + self._init_std) + self._min_std
      dist = torchd.normal.Normal(mean, std)
      dist = tools.ContDist(torchd.independent.Independent(dist, 1))
    elif self._dist == 'normal_1':
      x = self._dist_layer(x)
      dist = torchd.normal.Normal(mean, 1)
      dist = tools.ContDist(torchd.independent.Independent(dist, 1))
    elif self._dist == 'trunc_normal':
      x = self._dist_layer(x)
      mean, std = torch.split(x, [self._size]*2, -1)
      mean = torch.tanh(mean)
      std = 2 * torch.sigmoid(std / 2) + self._min_std
      dist = tools.SafeTruncatedNormal(mean, std, -1, 1)
      dist = tools.ContDist(torchd.independent.Independent(dist, 1))
    elif self._dist == 'onehot':
      x = self._dist_layer(x)
      dist = tools.OneHotDist(x)
    elif self._dist == 'onehot_gumble':
      x = self._dist_layer(x)
      temp = self._temp
      dist = tools.ContDist(torchd.gumbel.Gumbel(x, 1/temp))
    else:
      raise NotImplementedError(self._dist)
    return dist


class GRUCell(nn.Module):

  def __init__(self, inp_size,
               size, norm=False, act=torch.tanh, update_bias=-1):
    super(GRUCell, self).__init__()
    self._inp_size = inp_size
    self._size = size
    self._act = act
    self._norm = norm
    self._update_bias = update_bias
    self._layer = nn.Linear(inp_size+size, 3*size,
                            bias=norm is not None)
    if norm:
      self._norm = nn.LayerNorm(3*size)

  @property
  def state_size(self):
    return self._size

  def forward(self, inputs, state):
    state = state[0]  # Keras wraps the state in a list.
    parts = self._layer(torch.cat([inputs, state], -1))
    if self._norm:
      parts = self._norm(parts)
    reset, cand, update = torch.split(parts, [self._size]*3, -1)
    reset = torch.sigmoid(reset)
    cand = self._act(reset * cand)
    update = torch.sigmoid(update + self._update_bias)
    output = update * cand + (1 - update) * state
    return output, [output]

def build_mlp(
        input_size: int,
        output_size: int,
        n_layers: int,
        hidden_size: int,
        activation: Activation = 'tanh',
        output_activation: Activation = 'identity',
) -> nn.Module:
    """
        Builds a feedforward neural network

        arguments:
            n_layers: number of hidden layers
            size: dimension of each hidden layer
            activation: activation of each hidden layer

            input_size: size of the input layer
            output_size: size of the output layer
            output_activation: activation of the output layer

        returns:
            MLP (nn.Module)
    """
    if isinstance(activation, str):
        activation = _str_to_activation[activation.lower()]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]

    # TODO: return a MLP. This should be an instance of nn.Module
    # Note: nn.Sequential is an instance of nn.Module.
    layers = []
    assert n_layers >= 0, f"Num layers should be larger than or equal to zero"
    assert isinstance(n_layers, int), f"Num layers should be integer" 
    if n_layers == 0:
      layers.append(nn.Linear(input_size, output_size))
    else:
      layers.append(nn.Linear(input_size, hidden_size))
      for i in range(n_layers):
        layers.append(activation)
        if i == n_layers - 1:
            out = output_size
        else:
            out = hidden_size
        layers.append(nn.Linear(hidden_size, out))
    layers.append(output_activation)
    return nn.Sequential(*layers)
