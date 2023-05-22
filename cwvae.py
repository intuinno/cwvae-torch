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
        self.pre_layers = nn.ModuleList()
        self.optimizers = []

        
        if configs.dyn_discrete:
            feat_size = configs.cell_stoch_size * configs.dyn_discrete + configs.cell_deter_size
        else:
            feat_size = configs.cell_stoch_size + configs.cell_deter_size
        
        input_channels = configs.channels
        
        for level in range(configs.levels):
            layer = nn.ModuleDict()
            if level == 0:
                self.pre_layers.append(networks.preprocessAE())
                layer['encoder'] = networks.ConvEncoder(
                    configs.cell_embed_size,
                    channels=configs.channels,
                    depth=configs.cnn_depth,
                    act=getattr(nn,configs.act),
                    kernels=configs.encoder_kernels
                )
                shape = (*configs.img_size, input_channels)
                
                layer['decoder'] = networks.ConvDecoder(
                    feat_size, 
                    depth=configs.cnn_depth,
                    act=getattr(nn, configs.act),
                    shape=(configs.channels, *configs.img_size),
                    kernels=configs.decoder_kernels,
                    thin=configs.decoder_thin
                ) 
            else:
                if configs.pre_discrete:
                    discrete_factor = configs.pre_discrete
                else:
                    discrete_factor = 1
                    
                
                if level == 1:
                    emb_shape = (16,16,16 * discrete_factor)
                    hid_factor = discrete_factor
                elif level == 2:
                    emb_shape = (4, 4, 256 * discrete_factor)
                    hid_factor = 1
                else:
                    raise NotImplementedError
                pre_encoder = networks.Conv3dVAE(input_channels=input_channels, 
                                                 emb_shape=emb_shape,
                                                 hid_factor=hid_factor,
                                                 discrete=configs.pre_discrete)
                
                self.pre_layers.append(pre_encoder)
                
                
                H = shape[0] // 4**(level)
                W = shape[1] // 4**(level)
                C = shape[2] * 16**(level) 
                
                input_channels = C * discrete_factor
                
                layer['encoder'] = networks.LocalConvEncoder( configs.cell_embed_size,
                                                            channels_factor=2,
                                                            input_width=W,
                                                            input_height=H,
                                                            input_channels=input_channels,
                                                            )
                
                layer['decoder'] = networks.LocalConvDecoder(feat_size=feat_size,
                                                          shape=emb_shape)
                
                    
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
                embed=configs.cell_embed_size,
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
        if configs.levels > 1:
            self.pre_opt = tools.Optimizer(
                    f'preAE_opt',
                    self.pre_layers.parameters(),
                    configs.lr, 
                    eps=configs.eps, 
                    clip=configs.clip_grad_norm_by,
                    wd=configs.weight_decay,
                    opt=configs.optimizer,
                    use_amp=False
                ) 
        test = torch.rand((4,16,64,64,1))
        self.hierarchical_encode(test)
        
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
        B, _, _, _, _ = obs.shape
        input = obs

        for level in range(self._levels):
            pre_embedding = self.pre_layers[level].encode(input)
            pre_embedding = einops.rearrange(pre_embedding, 'b t h w c -> (b t) h w c')
            pre_embedding = pre_embedding.clone().detach().requires_grad_(False)
            embedding = self.layers[level]['encoder'](pre_embedding)
            pre_embedding = einops.rearrange(pre_embedding, '(b t) h w c -> b t h w c', b=B)
            embedding = einops.rearrange(embedding, '(b t) e -> b t e', b=B)
            outputs.append(embedding)
            recon_target.append(pre_embedding)

            # embedding is [B, T, H, W, C] dimension
            # To reduce with _tmp_abs_factor^level, we need to pad T dimension of embedding
            # such that it is divisible with timesteps_to_merge
            timesteps_to_merge = self._tmp_abs_factor
            timesteps_to_pad = np.mod(
                timesteps_to_merge - np.mod(embedding.shape[1], timesteps_to_merge),
                timesteps_to_merge
            )
            pad = (0, 0, 0, 0, 0, 0, 0, timesteps_to_pad, 0, 0)
            input = F.pad(pre_embedding, pad, "constant", 0)
            input = input.clone().detach().requires_grad_(False)
        return outputs, recon_target
        
    def hierarchical_observe(
        self, inputs, actions=None, initial_state=None
    ):
        B, T, _ = inputs[0].shape
        
        context = None # None for top level
        kl_balance = tools.schedule(self.configs.kl_balance, self.step)
        kl_free = tools.schedule(self.configs.kl_free, self.step)
        kl_scale = tools.schedule(self.configs.kl_scale, self.step)
        prior_list, posterior_list = [], []
        kl_loss_list, kl_value_list = [], []
        feat_list = []

        for level in reversed(range(self._levels)):
            if actions==None:
                empty_action = torch.empty(B, T, 0).to(self.device)
                actions = empty_action 
            inp = inputs[level]
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
        feat_list = []

        for level in reversed(range(self._levels)):
            num_steps = np.ceil(float(num_imagine)/(self._tmp_abs_factor**level))
            empty_action = torch.empty(batch_size, int(num_steps),0).to(self.device)
            prior = self.layers[level]['dynamics'].imagine(context, empty_action, initial_state[level])

            feat = self.layers[level]['dynamics'].get_feat(prior)
            feat_list.insert(0, feat)

            if self._discrete:
                stoch = einops.rearrange(prior['stoch'], 'b t d f -> b t (d f)')
            else:
                stoch = prior['stoch']
            context = torch.concat([prior['deter'], stoch], dim=-1) 
            context = einops.repeat(context, 'b t f -> b (t repeat) f', repeat=self._tmp_abs_factor)

        return feat_list
    
    def pre_decode(self, emb, level=0):
        dec = emb
        for i in reversed(range(level+1)):
            dec = self.pre_layers[i].decode(dec)
        return dec
        
    def hierarchical_pre_decode(self, feat_list, data):
        image_list = []
        recon_loss_list = []
        for level in range(self._levels):
            recon = self.layers[level]['decoder'](feat_list[level]).mode()
            recon_image = self.pre_decode(recon, level=level)
            recon_image = recon_image[:, :data.shape[1]] 
            mse = F.mse_loss( recon_image, data)
            mse = to_np(mse) 
            recon_loss_list.append(mse)
            image_list.append(recon_image) 
        return image_list, recon_loss_list

    def pred(self, data, num_initial=16, video_layer=0):
        b, t, c, w, h = data.shape
        num_imagine = t - num_initial
        obs = data[:,:num_initial]
        embed, _ = self.hierarchical_encode(obs)
        posteriors, _, _, _, feat_list = self.hierarchical_observe(embed) 

        initial_decode, _ = self.hierarchical_pre_decode(feat_list, obs)

        empty_action = torch.empty(b, num_imagine, 0).to(self.device)
        init_states = []
        for level in range(self._levels):
            init = {k: v[:,-1] for k, v in posteriors[level].items()}
            init_states.append(init)

        feat_list = self.hierarchical_imagine(empty_action, initial_state=init_states)
        openl, recon_loss_list = self.hierarchical_pre_decode(feat_list, data[:, num_initial:])
        
        return openl, recon_loss_list, initial_decode
        
    def video_pred(self, data, video_layer=0):
        num_initial = self._tmp_abs_factor ** (self._levels-1)        
        openl, recon_loss, initial_decode = self.pred(data, num_initial=num_initial, video_layer=0)
        num_gifs = 6
        data = self.pre_layers[0].encode(data)
        truth = self.pre_layers[0].decode(data[:num_gifs])  
        openl = torch.cat(openl, 2)
        initial_decode = torch.cat(initial_decode,2)
        model = torch.cat([initial_decode,  openl], 1)[:num_gifs]
        return_video = torch.cat([truth, model], 2) 
        # return_video = (return_video * 255).to(dtype=torch.uint8)
        return to_np(return_video), recon_loss
    

    def local_train(self, obs, stop_level):
        metrics = {}
        with tools.RequiresGrad(self):
            with torch.cuda.amp.autocast(self._use_amp):
                embed, recon_target = self.hierarchical_encode(obs)
                posteriors, priors, kl_losses, kl_values, feats = self.hierarchical_observe(embed)

                for level in reversed(range(self._levels-1, stop_level-1, -1)):
                    #Calulate reconstruction loss
                    pred_obs = self.layers[level]['decoder'](feats[level])
                    nll = -pred_obs.log_prob(recon_target[level])
                    recon_loss = nll.sum()
                    kl_loss = kl_losses[level]       
                    loss = kl_loss + recon_loss
                    if DEBUG:
                        dot = make_dot(loss, params=dict(self.named_parameters()))
                        dot.render(f"loss_graph_{level}.pdf")
                    metrics[f'recon_loss_{level}'] = to_np(recon_loss)
                    metrics[f'kl_loss_{level}'] = to_np(kl_loss)
                    metrics[f'loss_{level}'] = to_np(loss)
                    metrics[f'grad_norm_{level}'] = self.optimizers[level](loss, self.layers[level].parameters())
                    metrics[f'kl_{level}'] = to_np(torch.mean(kl_values[level]))
                    metrics[f'prior_ent_{level}'] = to_np(torch.mean(self.layers[level]['dynamics'].get_dist(priors[level]).entropy()))
                    metrics[f'posterior_ent_{level}'] = to_np(torch.mean(self.layers[level]['dynamics'].get_dist(posteriors[level]).entropy()))

        return metrics

    def pre_train(self, obs, train_level=2):
        metrics = {}
        with tools.RequiresGrad(self):
            # with torch.cuda.amp.autocast(self._use_amp):

            recons, embeddings, recon_targets, dists = self.hierarchical_pre_encode(obs)
            
            loss = 0 
            for level in range(1, train_level):
                recon_loss = F.mse_loss(recons[level], recon_targets[level],reduction='sum')
                kl_loss = self.pre_layers[level].kl_divergence(dists[level])
                kl_loss = kl_loss.sum()
                loss += recon_loss + kl_loss
                metrics[f'pre_recon_loss_{level}'] = to_np(recon_loss)
                metrics[f'pre_kl_loss_{level}'] = to_np(kl_loss)
                
            metrics[f'pre_grad_norm_pre_encder'] = self.pre_opt(loss, self.pre_layers.parameters())
            if DEBUG:
                dot = make_dot(recon_loss, params=dict(self.pre_layers.named_parameters()))
                dot.render(f"graph_{level}.pdf")
        return metrics
                    
    def pre_eval(self, data):
        _, T, _, _, _ = data.shape
        with torch.no_grad():
            recons, embeddings, recon_targets, _ = self.hierarchical_pre_encode(data)
            recon_loss_list = []
            for level in range(1, self._levels):
                recon_loss = F.mse_loss(recons[level], recon_targets[level])
                loss = recon_loss
                recon_loss_list.append(to_np(loss)) 
        num_gifs = 6
        truth = recons[0][:num_gifs]  
        layer1_recon = self.pre_decode(embeddings[1], level=1)[:num_gifs, :T ]
        layer2_recon = self.pre_decode(embeddings[2], level=2)[:num_gifs, :T] 
        return_video = torch.cat([truth, layer1_recon, layer2_recon], 2)
        # return_video = (return_video * 255).to(dtype=torch.uint8)
        return to_np(return_video), recon_loss_list

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
        for level in range(0, self._levels):
            recon_targets.append(obs) 
            recon, dist, embedding = self.pre_layers[level].forward(obs)
            recons.append(recon)
            dists.append(dist)
            

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
           
            embeddings.append(embedding)
           
        return recons, embeddings, recon_targets, dists
        

                
        

