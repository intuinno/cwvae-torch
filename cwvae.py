from torch import nn
import torch
import networks
import tools
import numpy as np

to_np = lambda x: x.detach().cpu().numpy()

device = "cuda" if torch.cuda.is_available() else "cpu"


class CWVAE(nn.Module):
    
    def __init__(self, configs):
        super(CWVAE, self).__init__()
        self.step = 0
        self._use_amp = True if configs.precision==16 else False
        self.encoder = networks.ConvEncoder(channels=configs.channels, 
                                            depth=configs.cnn_depth,
                                            act=getattr(nn,configs.act),
                                            kernels=configs.encoder_kernels)

        shape = (*configs.img_size, configs.channels)
        # Get the output embedding size of ConvEncoder by testing it
        testObs = torch.rand(1, 1, *shape) 
        embed_size = self.encoder(testObs).shape[-1]
        self.configs = configs
        
        self.dynamics = networks.RSSM(
            stoch=configs.cell_stoch_size,
            deter=configs.cell_deter_size,
            hidden=configs.cell_deter_size,
            layers_input=configs.dyn_input_layers,
            layers_output=configs.dyn_output_layers,
            rec_depth=configs.dyn_rec_depth,
            shared=configs.dyn_shared, 
            discrete=configs.dyn_discrete,
            act=getattr(nn, configs.act),
            mean_act=configs.dyn_mean_act,
            std_act=configs.dyn_std_act,
            temp_post=configs.dyn_temp_post,
            min_std=configs.cell_min_stddev,
            cell=configs.dyn_cell,
            num_actions=0,
            embed=embed_size,
            device=configs.device
        )
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
        
    def train(self, obs):
        # obs = self.preprocess(obs)
        
        
        with tools.RequiresGrad(self):
            with torch.cuda.amp.autocast(self._use_amp):
                obs = self.preprocess(obs)
                embed = self.encoder(obs)
                b, t, f = embed.shape
                empty_action = torch.empty(b, t, 0).to(device)
                post, prior = self.dynamics.observe(embed, empty_action)
                kl_balance = tools.schedule(self.configs.kl_balance, self.step)
                kl_free = tools.schedule(self.configs.kl_free, self.step)
                kl_scale = tools.schedule(self.configs.kl_scale, self.step)
                kl_loss, kl_value = self.dynamics.kl_loss(
                    post, prior, self.configs.kl_forward, kl_balance, kl_free, kl_scale)
                feat = self.dynamics.get_feat(post)
                pred_obs = self.decoder(feat)
                nll = -pred_obs.log_prob(obs)
                recon_loss = nll.mean()
                loss = kl_loss + recon_loss
            metrics = self.opt(loss, self.parameters())
            
        metrics['kl_balance'] = kl_balance
        metrics['kl_free'] = kl_free
        metrics['kl_scale'] = kl_scale
        metrics['kl'] = to_np(torch.mean(kl_value))
        
        with torch.cuda.amp.autocast(self._use_amp):
            metrics['prior_ent'] = to_np(torch.mean(self.dynamics.get_dist(prior).entropy()))
            metrics['posterior_ent'] = to_np(torch.mean(self.dynamics.get_dist(post).entropy()))
            context = dict(
                embed=embed,
                feat=self.dynamics.get_feat(post),
                kl=kl_value,
                postent=self.dynamics.get_dist(post).entropy()
            )
        post = {k: v.detach() for k, v in post.items()}
        return post, context, metrics
        
    def video_pred(self, data):
        b, t, c, w, h = data.shape
        num_initial = 5
        num_gifs = 6
        data = self.preprocess(data)
        truth = data[:num_gifs] + 0.5 
        embed = self.encoder(data)
        empty_action = torch.empty(b, t, 0).to(device)
        
        post, _ = self.dynamics.observe(embed[:, :num_initial], empty_action[:,:num_initial])
        initial_decode = self.decoder(self.dynamics.get_feat(post)).mode()[:num_gifs]
        init = {k: v[:,-1] for k, v in post.items()}
        prior = self.dynamics.imagine(empty_action[:, num_initial:], init)
        
        feat = self.dynamics.get_feat(prior)
        pred_obs = self.decoder(feat)
        nll = -pred_obs.log_prob(data[:, num_initial:])
        recon_loss = nll.mean()
        openl = self.decoder(feat).mode()[:num_gifs]
        model = torch.cat([initial_decode + 0.5,  openl + 0.5], 1)
        diff = (model - truth + 1) / 2
        return_video = torch.cat([truth, model, diff], 2) 
        # return_video = (return_video * 255).to(dtype=torch.uint8)
        return to_np(return_video), recon_loss
        
    def preprocess(self, obs):
        obs = obs.clone()
        obs = obs / 255.0 - 0.5
        obs.to(device)
        return obs
                
        
        