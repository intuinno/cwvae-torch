from torch import nn
import torch
import networks
import tools
import numpy as np
import einops
import torch.nn.functional as F
from torch import distributions as torchd




DEBUG = False

if DEBUG:
    from torchview import draw_graph
    import graphviz
    # graphviz.set_jupyter_format('png')
    from torchviz import make_dot, make_dot_from_trace

to_np = lambda x: x.detach().cpu().numpy()

# device = "cuda" if torch.cuda.is_available() else "cpu"

class CWVAE(nn.Module):
    
    def __init__(self, configs):
        super(CWVAE, self).__init__()
        self.step = 0
        self._use_amp = True if configs.precision==16 else False
        self.layers = nn.ModuleList()
        self.optimizers = []
        self.pre_opt = {}
        self.pre_layers = nn.ModuleDict()
        
        if configs.dyn_discrete:
            feat_size = configs.cell_stoch_size * configs.dyn_discrete + configs.cell_deter_size
        else:
            feat_size = configs.cell_stoch_size + configs.cell_deter_size
        
        for level in range(configs.levels):
            layer = nn.ModuleDict()
            if level == 0:
                layer['encoder'] = networks.ConvEncoder(
                    channels=configs.channels,
                    depth=configs.cnn_depth,
                    act=getattr(nn,configs.act),
                    kernels=configs.encoder_kernels
                )
                shape = (*configs.img_size, configs.channels)
                # Get the output embedding size of ConvEncoder by testing it
                testObs = torch.rand(1, 1, *shape) 
                testEmbed = layer['encoder'](testObs)
                
                embed_shape = layer['encoder'](testObs).shape
                _, C, _, H, W = embed_shape
                embed_size = embed_shape.numel()
                
                layer['decoder'] = networks.ConvDecoder(
                    feat_size, 
                    depth=configs.cnn_depth,
                    act=getattr(nn, configs.act),
                    shape=(configs.channels, *configs.img_size),
                    kernels=configs.decoder_kernels,
                    thin=configs.decoder_thin
                ) 
            else:
                layer['encoder'] = networks.Conv3dEncoder()
                layer['decoder'] = networks.Conv3dDecoder(feat_size=feat_size,
                                                          shape=(C, H, W))
                input_channels = configs.channels * (2*configs.tmp_abs_factor) ** (level-1)
                self.pre_layers[str(level)] = networks.Conv3dVAE(input_channels=input_channels)
                
                opt = tools.Optimizer(
                    f'preAE_{level}',
                    self.pre_layers[str(level)].parameters(),
                    configs.lr, 
                    eps=configs.eps, 
                    clip=configs.clip_grad_norm_by,
                    wd=configs.weight_decay,
                    opt=configs.optimizer,
                    use_amp=self._use_amp
                )
                self.pre_opt[str(level)] = opt
                    
            layer['dynamics'] = networks.RSSM(
                stoch=configs.cell_stoch_size,
                deter=configs.cell_deter_size,
                hidden=configs.cell_deter_size,
                layers_input=configs.dyn_input_layers,
                layers_output=configs.dyn_output_layers,
                discrete=configs.dyn_discrete,
                act=getattr(nn, configs.act),
                mean_act=configs.dyn_mean_act,
                std_act=configs.dyn_std_act,
                min_std=configs.cell_min_stddev,
                cell=configs.dyn_cell,
                num_actions=0,
                embed=embed_size,
                device=configs.device
            )
                
            self.layers.append(layer)
            opt = tools.Optimizer(
                f'level_{level}',
                layer.parameters(),
                configs.lr, 
                eps=configs.eps, 
                clip=configs.clip_grad_norm_by,
                wd=configs.weight_decay,
                opt=configs.optimizer,
                use_amp=self._use_amp
            )
            self.optimizers.append(opt)
            
        self.configs = configs
        self.device = configs.device
        self._levels = configs.levels
        self._tmp_abs_factor = configs.tmp_abs_factor
        self._discrete = configs.dyn_discrete
        self.pre_loss = nn.MSELoss()
        
    def hierarchical_encode(self,
                            obs):
        """
        Arguments:
            obs : Tensor
                Un-flattened observations (videos) of shape BTHWC (batch size, timesteps, height, width, channel)
        Return:
            List of Tensors
                Each element is Un-flattened observations (videos) of shape BTHWC for 
                (batch size, timesteps, height, width, channel ) 
    """ 
        outputs = []
        recon_target = []

        for level in range(self._levels):
           
            embedding = self.layers[level]['encoder'](obs)
            out = einops.rearrange(embedding, 'b c t h w -> b t h w c')
            outputs.append(out)
            if level == 0:
                target = obs
            else:
                target = einops.rearrange(obs.clone().detach().requires_grad_(False), 'b c t h w -> b t h w c')
            recon_target.append(target)

            # embedding is [B, C, T, H, W] dimension
            # To reduce with _tmp_abs_factor^level, we need to pad T dimension of embedding
            # such that it is divisible with timesteps_to_merge
            timesteps_to_merge = self._tmp_abs_factor
            timesteps_to_pad = np.mod(
                timesteps_to_merge - np.mod(embedding.shape[2], timesteps_to_merge),
                timesteps_to_merge
            )
            pad = (0, 0, 0, 0, 0, timesteps_to_pad, 0, 0, 0, 0)
            embedding = F.pad(embedding, pad, "constant", 0)
            obs = embedding.clone().detach().requires_grad_(False)
            
        recon_target

        return outputs, recon_target
        
    def hierarchical_observe(
        self, inputs, actions=None, initial_state=None
    ):
        b, t, h, w, c = inputs[0].shape
        

            
        context = None # None for top level
        kl_balance = tools.schedule(self.configs.kl_balance, self.step)
        kl_free = tools.schedule(self.configs.kl_free, self.step)
        kl_scale = tools.schedule(self.configs.kl_scale, self.step)
        prior_list, posterior_list = [], []
        kl_loss_list, kl_value_list = [], []
        feat_list = []

        for level in reversed(range(self._levels)):
            if actions==None:
                empty_action = torch.empty(b, t, 0).to(self.device)
                actions = empty_action 
            inp = einops.rearrange(inputs[level], 'b t h w c -> b t (h w c)')
            post, prior = self.layers[level]['dynamics'].observe(inp, context, actions)
            kl_loss, kl_value = self.layers[level]['dynamics'].kl_loss(
                post, prior, self.configs.kl_forward, kl_balance, kl_free, kl_scale)
            prior_list.insert(0, prior)
            posterior_list.insert(0, post)
            kl_loss_list.insert(0, kl_loss)
            kl_value_list.insert(0, kl_value)
            feat = self.layers[level]['dynamics'].get_feat(post)
            feat_list.insert(0, feat)

            # Build context for lower layer
            if self._discrete:
                stoch = einops.rearrange(post['stoch'], 'b t d f -> b t (d f)')
            else:
                stoch = post['stoch']
            context = torch.concat([post['deter'], stoch], dim=-1)
            context = einops.repeat(context, 'b t f -> b (t repeat) f', repeat=self._tmp_abs_factor)
            context = context.clone().detach().requires_grad_(False)
            
        return posterior_list, prior_list, kl_loss_list, kl_value_list, feat_list
    
    def hierarchical_imagine(
        self, actions=None, initial_state=None
    ):
        assert len(actions.shape) > 2, "actions need to be [B T A] shape. It is used to calculate total imagine steps"
        batch_size, num_imagine, _ = actions.shape
        context = None # None for top level

        for level in reversed(range(self._levels)):
            num_steps = np.ceil(float(num_imagine)/(self._tmp_abs_factor**level))
            empty_action = torch.empty(batch_size, int(num_steps),0).to(self.device)
            prior = self.layers[level]['dynamics'].imagine(context, empty_action, initial_state[level])

            if self._discrete:
                stoch = einops.rearrange(prior['stoch'], 'b t d f -> b t (d f)')
            else:
                stoch = prior['stoch']
            context = torch.concat([prior['deter'], stoch], dim=-1) 
            context = einops.repeat(context, 'b t f -> b (t repeat) f', repeat=self._tmp_abs_factor)

        # Get features for bottom layer
        feat = self.layers[0]['dynamics'].get_feat(prior)
            
        return feat

    def pred(self, data, num_initial=16):
        b, t, c, w, h = data.shape
        num_imagine = t - num_initial
        data = self.preprocess(data)
        truth = data + 0.5 
        obs = data[:,:num_initial]
        embed, _ = self.hierarchical_encode(obs)
        posteriors, _, _, _, feats = self.hierarchical_observe(embed) 
         
        initial_decode = self.layers[0]['decoder'](feats[0]).mode() + 0.5

        empty_action = torch.empty(b, num_imagine, 0).to(self.device)
        init_states = []
        for level in range(self._levels):
            init = {k: v[:,-1] for k, v in posteriors[level].items()}
            init_states.append(init)

        feat = self.hierarchical_imagine(empty_action, initial_state=init_states)
        pred_obs = self.layers[0]['decoder'](feat)
        nll = -pred_obs.log_prob(data[:, num_initial:])
        recon_loss = nll.mean()
        openl = self.layers[0]['decoder'](feat).mode() + 0.5
        openl = np.clip(to_np(openl), 0, 1)
        return openl, recon_loss, initial_decode

        
    def video_pred(self, data):
        num_initial = self._tmp_abs_factor ** (self._levels-1)        
        openl, recon_loss, initial_decode = self.pred(data, num_initial=num_initial)
        num_gifs = 6
        data = self.preprocess(data)
        truth = data[:num_gifs] + 0.5 
        openl = torch.Tensor(openl).to(self.device)
        model = torch.cat([initial_decode,  openl], 1)[:num_gifs]
        diff = (model - truth + 1) / 2
        return_video = torch.cat([truth, model, diff], 2) 
        # return_video = (return_video * 255).to(dtype=torch.uint8)
        return to_np(return_video), recon_loss
    

    def local_train(self, obs):
        
        metrics = {}
        with tools.RequiresGrad(self):
            with torch.cuda.amp.autocast(self._use_amp):
                obs = self.preprocess(obs)
                embed, recon_target = self.hierarchical_encode(obs)
                posteriors, priors, kl_losses, kl_values, feats = self.hierarchical_observe(embed)

                for level in range(self._levels):
                    #Calulate reconstruction loss
                    pred_obs = self.layers[level]['decoder'](feats[level])
                    nll = -pred_obs.log_prob(recon_target[level])
                    recon_loss = nll.mean()
                    kl_loss = kl_losses[level]       
                    loss = kl_loss + recon_loss
                    metrics[f'recon_loss_{level}'] = to_np(recon_loss)
                    metrics[f'kl_loss_{level}'] = to_np(kl_loss)
                    metrics[f'loss_{level}'] = to_np(loss)
                    metrics[f'grad_norm_{level}'] = self.optimizers[level](loss, self.layers[level].parameters())
                    metrics[f'kl_{level}'] = to_np(torch.mean(kl_values[level]))
                    metrics[f'prior_ent_{level}'] = to_np(torch.mean(self.layers[level]['dynamics'].get_dist(priors[level]).entropy()))
                    metrics[f'posterior_ent_{level}'] = to_np(torch.mean(self.layers[level]['dynamics'].get_dist(posteriors[level]).entropy()))

        return metrics

    def pre_train(self, obs):
        metrics = {}
        with tools.RequiresGrad(self):
            with torch.cuda.amp.autocast(self._use_amp):

                recons, embeddings, recon_targets = self.hierarchical_pre_encode(obs)
                
                for level in range(1, self._levels):
                    # recon_loss = F.binary_cross_entropy(recons[level-1], recon_targets[level-1], reduction = 'sum')
                    recon_loss = self.pre_loss(recons[level-1], recon_targets[level-1])
                    loss = recon_loss 
                    # self.pre_opt[str(level)].zero_grad()
                    
                    metrics[f'pre_grad_norm_{level}'] = self.pre_opt[str(level)](loss, self.pre_layers[str(level)].parameters())
                    if DEBUG:
                        dot = make_dot(recon_loss, params=dict(self.pre_layers.named_parameters()))
                        dot.render(f"graph_{level}.pdf")
                    # loss.backward()
                    # self.pre_opt[str(level)].step()
                    metrics[f'recon_loss_{level}'] = to_np(recon_loss)
                    # metrics[f'kl_loss_{level}'] = to_np(kl_loss)
                    metrics[f'loss_{level}'] = to_np(loss)
        return metrics
                    
    def pre_eval(self, data):
        recons, embeddings, recon_targets = self.hierarchical_pre_encode(data)
        for level in range(1, self._levels):
            recon_loss = F.binary_cross_entropy(recons[level-1], recon_targets[level-1], reduction = 'sum')
            loss = recon_loss 
            # metrics[f'recon_loss_{level}'] = to_np(recon_loss)
            # metrics[f'loss_{level}'] = to_np(loss)
        num_gifs = 6
        truth = recon_targets[0][:num_gifs]  
        layer1_recon = recons[0][:num_gifs] 
        layer2_recon = self.pre_layers['1'].decode(recons[1][:num_gifs]) 
        layer2_recon = layer2_recon[:, :truth.shape[1], :, :, :]
        return_video = torch.cat([truth, layer1_recon, layer2_recon], 2)
        # return_video = (return_video * 255).to(dtype=torch.uint8)
        return to_np(return_video), recon_loss
                      
        
    def pred(self, data, num_initial=16):
        b, t, c, w, h = data.shape
        num_imagine = t - num_initial
        data = self.preprocess(data)
        truth = data + 0.5 
        obs = data[:,:num_initial]
        embed, _ = self.hierarchical_encode(obs)
        posteriors, _, _, _, feats = self.hierarchical_observe(embed) 
         
        initial_decode = self.layers[0]['decoder'](feats[0]).mode() + 0.5

        empty_action = torch.empty(b, num_imagine, 0).to(self.device)
        init_states = []
        for level in range(self._levels):
            init = {k: v[:,-1] for k, v in posteriors[level].items()}
            init_states.append(init)

        feat = self.hierarchical_imagine(empty_action, initial_state=init_states)
        pred_obs = self.layers[0]['decoder'](feat)
        nll = -pred_obs.log_prob(data[:, num_initial:])
        recon_loss = nll.mean()
        openl = self.layers[0]['decoder'](feat).mode() + 0.5
        openl = np.clip(to_np(openl), 0, 1)
        return openl, recon_loss, initial_decode

    def hierarchical_pre_encode(self,
                            obs):
        """
        Arguments:
            obs : Tensor
                Un-flattened observations (videos) of shape BTHWC (batch size, timesteps, height, width, channel)
        Return:
            List of Tensors
                Each element is Un-flattened observations (videos) of shape BTHWC for 
                (batch size, timesteps, height, width, channel ) 
    """ 
    
                    
        embeddings, recons, recon_targets, dists = [], [], [], []
        embedding = self.preprocess(obs) 
        for level in range(1, self._levels):
            # embedding is [B, T, H, W, C] dimension
            # To reduce with _tmp_abs_factor^level, we need to pad T dimension of embedding
            # such that it is divisible with timesteps_to_merge
            timesteps_to_merge = self._tmp_abs_factor
            timesteps_to_pad = np.mod(
                timesteps_to_merge - np.mod(embedding.shape[1], timesteps_to_merge),
                timesteps_to_merge
            )
            pad = (0, 0, 0, 0, 0, 0, 0, timesteps_to_pad, 0, 0)
            embedding = F.pad(embedding, pad, "constant", 0)
            obs = embedding.clone().detach().requires_grad_(False)
           
            recon, embedding = self.pre_layers[str(level)].forward(obs)
            recons.append(recon+0.5)
            embeddings.append(embedding)
            recon_targets.append(obs+0.5) 

        return recons, embeddings, recon_targets
        
    def preprocess(self, obs):
        # obs = obs.clone()
        obs = obs / 255.0 - 0.5
        obs.to(self.device)
        return obs
                
        

