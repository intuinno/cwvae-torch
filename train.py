import numpy as np
import argparse
import os
import ruamel.yaml as yaml
import torch
import pathlib
import sys
from tqdm import tqdm

# from cwvae import build_model
# from loggers.summary import Summary
# from loggers.checkpoint import Checkpoint
from data_loader import *
import tools
from cwvae import CWVAE
from datetime import datetime
import pytz


# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using {device} device")




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs='+', required=True)
    args, remaining = parser.parse_known_args()
    rootdir = pathlib.Path(sys.argv[0]).parent
    configs = yaml.safe_load((rootdir / 'configs.yml').read_text())
    
    defaults = {}
    for name in args.configs:
        defaults.update(configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f'--{key}', type=arg_type, default=arg_type(value))
    configs = parser.parse_args(remaining)

    exp_name = configs.exp_name + '_'

    
    tz = pytz.timezone("US/Central")
    now = datetime.now(tz)
    date_time = now.strftime("%Y%m%d_%H%M%S")
    exp_name += date_time
    
    # Creating model dir with experiment name.
    exp_logdir = rootdir / configs.logdir / configs.dataset / exp_name
    print('Logdir', exp_logdir)
    exp_logdir.mkdir(parents=True, exist_ok=True)

    # Dumping config.
    with open(exp_logdir / "config.yml", "w") as f:
        yaml.dump(configs, f, default_flow_style=False)

    # Load dataset.
    train_dataloader, val_dataloader = load_dataset(configs)

    # Build model
    model = CWVAE(configs).to(configs.device)
    print(f"========== Using {configs.device} device ===================")

    # Build logger
    logger = tools.Logger(exp_logdir, 0)
    metrics = {}

    for epoch in range(configs.num_epochs):
            
        #Write evaluation summary
        print(f'======== Epoch {epoch} / {configs.num_epochs} ==========')
        now = datetime.now(tz)
        current_time = now.strftime("%H:%M:%S")
        print("Current Time =", current_time)
        print (f"Evaluating ... ") 
        logger.step = epoch
        if epoch % configs.eval_every == 0:
            x = next(iter(val_dataloader))
            openl, recon_loss = model.video_pred(x.to(configs.device))
            logger.video('eval_openl', openl)
            logger.scalar('eval_video_nll', recon_loss)
            logger.write(fps=True)
        
        print(f"Training ...")
        for i, x in enumerate(tqdm(train_dataloader)):
            x = x.to(configs.device)
            met = model.train(x)
            for name, values in met.items():
                if not name in metrics.keys():
                    metrics[name] = [values]
                else:
                    metrics[name].append(values)
        
        # Write training summary 
        for name,values in metrics.items():
            logger.scalar(name, float(np.mean(values)))
            metrics[name] = [] 
        openl, recon_loss = model.video_pred(x)
        logger.video('train_openl', openl)
        logger.write(fps=True)
            
        # Save Check point
        if epoch % configs.save_model_every == 0:
            torch.save(model.state_dict(), exp_logdir / 'latest_model.pt')

    print("Training complete.")
