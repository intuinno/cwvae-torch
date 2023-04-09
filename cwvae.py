from torch import nn
import torch
import networks
import tools
import numpy as np
import einops

to_np = lambda x: x.detach().cpu().numpy()

# device = "cuda" if torch.cuda.is_available() else "cpu"


class CWVAE(nn.Module):
    
    def __init__(self, configs):
        super(CWVAE, self).__init__()
        self.step = 0
        self._use_amp = True if configs.precision==16 else False
        self.encoder = networks.HierarchicalEncoder(configs.levels,
                                                    configs.tmp_abs_factor,
                                                    configs.enc_dense_layers,
                                                    hidden_size=configs.enc_dense_hidden_size,
                                                    channels=configs.channels, 
                                                    depth=configs.cnn_depth,
                                                    act=configs.act,
                                                    kernels=configs.encoder_kernels,
                                                    input_size=configs.img_size)

        embed_size = self.encoder.embed_size 
        self.configs = configs

        self.dynamics = nn.ModuleList()
        self.dense_encoders = nn.ModuleList()

        
        for i in range(configs.levels):
            dynamic = networks.RSSM(
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
            self.dynamics.append(dynamic)

            
        if configs.dyn_discrete:
            feat_size = configs.cell_stoch_size * configs.dyn_discrete + configs.cell_deter_size
        else:
            feat_size = configs.cell_stoch_size + configs.cell_deter_size
        self.decoder = networks.ConvDecoder(
            feat_size,
            depth=configs.cnn_depth,
            act=getattr(nn,configs.act),
            shape=(configs.channels, *configs.img_size),
            kernels=configs.decoder_kernels,
            thin=configs.decoder_thin
        )
        self.opt = tools.Optimizer(
            'model',
            self.parameters(),
            configs.lr,
            eps=configs.eps,
            clip=configs.clip_grad_norm_by,
            wd=configs.weight_decay,
            opt=configs.optimizer,
            use_amp=self._use_amp
        )
        self.device = configs.device
        self._levels = configs.levels
        self._tmp_abs_factor = configs.tmp_abs_factor
        self._discrete = configs.dyn_discrete
        
    def hierarchical_observe(
        self, inputs, actions=None, initial_state=None
    ):
        b, t, f = inputs[0].shape
        
        if actions==None:
            empty_action = torch.empty(b, t, 0).to(self.device)
            actions = empty_action 
            
        context = None # None for top level
        kl_balance = tools.schedule(self.configs.kl_balance, self.step)
        kl_free = tools.schedule(self.configs.kl_free, self.step)
        kl_scale = tools.schedule(self.configs.kl_scale, self.step)
        prior_list, posterior_list = [], []
        kl_loss_list, kl_value_list = [], []
        kl_losses = 0

        for level in reversed(range(self._levels)):
            post, prior = self.dynamics[level].observe(inputs[level], context, actions)
            kl_loss, kl_value = self.dynamics[level].kl_loss(
                post, prior, self.configs.kl_forward, kl_balance, kl_free, kl_scale)
            prior_list.insert(0, prior)
            posterior_list.insert(0, post)
            kl_loss_list.append(kl_loss)
            kl_losses += kl_loss
            kl_value_list.append(kl_value)

            # Build context for lower layer
            if self._discrete:
                stoch = einops.rearrange(post['stoch'], 'b t d f -> b t (d f)')
            else:
                stoch = post['stoch']
            context = torch.concat([post['deter'], stoch], dim=-1)
            context = einops.repeat(context, 'b t f -> b (t repeat) f', repeat=self._tmp_abs_factor)

        # Get features for bottom layer
        feat = self.dynamics[0].get_feat(post)
            
        return posterior_list, prior_list, kl_losses, kl_value_list, feat
    
    def hierarchical_imagine(
        self, actions=None, initial_state=None
    ):
        assert len(actions.shape) > 2, "actions need to be [B T A] shape. It is used to calculate total imagine steps"
        batch_size, num_imagine, _ = actions.shape
        context = None # None for top level

        for level in reversed(range(self._levels)):
            num_steps = np.ceil(float(num_imagine)/(self._tmp_abs_factor**level))
            empty_action = torch.empty(batch_size, int(num_steps),0).to(self.device)
            prior = self.dynamics[level].imagine(context, empty_action, initial_state[level])

            if self._discrete:
                stoch = einops.rearrange(prior['stoch'], 'b t d f -> b t (d f)')
            else:
                stoch = prior['stoch']
            context = torch.concat([prior['deter'], stoch], dim=-1) 
            context = einops.repeat(context, 'b t f -> b (t repeat) f', repeat=self._tmp_abs_factor)

        # Get features for bottom layer
        feat = self.dynamics[0].get_feat(prior)
            
        return feat

        
    def video_pred(self, data):
        b, t, c, w, h = data.shape
        num_initial = self._tmp_abs_factor ** (self._levels-1)
        num_imagine = t - num_initial
        num_gifs = 6
        data = self.preprocess(data)
        truth = data[:num_gifs] + 0.5 
        embed = self.encoder(data[:,:num_initial])
        posteriors, _, _, _, feat = self.hierarchical_observe(embed) 
         
        initial_decode = self.decoder(feat).mode()[:num_gifs]

        empty_action = torch.empty(b, num_imagine, 0).to(self.device)
        init_states = []
        for level in range(self._levels):
            init = {k: v[:,-1] for k, v in posteriors[level].items()}
            init_states.append(init)

        feat = self.hierarchical_imagine(empty_action, initial_state=init_states)
        pred_obs = self.decoder(feat)
        nll = -pred_obs.log_prob(data[:, num_initial:])
        recon_loss = nll.mean()
        openl = self.decoder(feat).mode()[:num_gifs]
        model = torch.cat([initial_decode + 0.5,  openl + 0.5], 1)
        diff = (model - truth + 1) / 2
        return_video = torch.cat([truth, model, diff], 2) 
        # return_video = (return_video * 255).to(dtype=torch.uint8)
        return to_np(return_video), recon_loss
        
    def train(self, obs):
        
        with tools.RequiresGrad(self):
            with torch.cuda.amp.autocast(self._use_amp):
                obs = self.preprocess(obs)
                embed = self.encoder(obs)
                posteriors, priors, kl_losses, kl_values, feat = self.hierarchical_observe(embed)
                #Calulate reconstruction loss
                pred_obs = self.decoder(feat)
                nll = -pred_obs.log_prob(obs)
                recon_loss = nll.mean()
                
                loss = kl_losses + recon_loss
            metrics = self.opt(loss, self.parameters())

        with torch.cuda.amp.autocast(self._use_amp):
            for level in range(self._levels):
                metrics[f'kl_{level}'] = to_np(torch.mean(kl_values[level]))
                metrics[f'prior_ent_{level}'] = to_np(torch.mean(self.dynamics[level].get_dist(priors[level]).entropy()))
                metrics[f'posterior_ent_{level}'] = to_np(torch.mean(self.dynamics[level].get_dist(posteriors[level]).entropy()))

        return metrics
        

        
    def preprocess(self, obs):
        # obs = obs.clone()
        obs = obs / 255.0 - 0.5
        obs.to(self.device)
        return obs
                
        
        