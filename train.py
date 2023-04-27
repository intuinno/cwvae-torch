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
from prettytable import PrettyTable

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using {device} device")

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
    

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    
      
def main(rank: int, world_size: int, configs):
    ddp_setup(rank, world_size)

    if rank == 0:
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

        # Build logger
        logger = tools.Logger(exp_logdir, 0)
        metrics = {}   
            

    # Load dataset.
    train_dataset, val_dataset = load_dataset(configs)
    train_dataloader = DataLoader(train_dataset, batch_size=configs.batch_size, 
                                  shuffle=True, pin_memory=True, sampler=DistributedSampler(train_dataset) )
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=True)

    # Build model
    model = CWVAE(configs).to(rank)
    model = DDP(model, device_ids=[rank])
    count_parameters(model)
    
    print(f"========== Using {rank} device ===================")



    for epoch in range(configs.num_epochs):

        if rank == 0:
                
            #Write evaluation summary
            print(f'======== Epoch {epoch} / {configs.num_epochs} ==========')
            now = datetime.now(tz)
            current_time = now.strftime("%H:%M:%S")
            print("Current Time =", current_time)
            print (f"Evaluating ... ") 
            logger.step = epoch
            if epoch % configs.eval_every == 0:
                x = next(iter(val_dataloader))
                openl, recon_loss = model.video_pred(x.to(rank))
                logger.video('eval_openl', openl)
                logger.scalar('eval_video_nll', recon_loss)
                logger.write(fps=True)
            
        print(f"Training ...")
        
        if rank == 0:
            dl = tqdm(train_dataloader)
        else:
            dl = train_dataloader
        for i, x in enumerate(dl):
            x = x.to(rank)
            met = model.train(x)
            
            if rank == 0:
                for name, values in met.items():
                    if not name in metrics.keys():
                        metrics[name] = [values]
                    else:
                        metrics[name].append(values)
        
        # Write training summary 
        if rank == 0:
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

    destroy_process_group()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs='+', required=True)
    args, remaining = parser.parse_known_args()
    rootdir = pathlib.Path(sys.argv[0]).parent
    configs = yaml.safe_load((rootdir / 'configs.yml').read_text())
    
    defaults = {}
    exp_name = ""
    for name in args.configs:
        defaults.update(configs[name])
        exp_name = exp_name + name + '_'
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f'--{key}', type=arg_type, default=arg_type(value))
    configs = parser.parse_args(remaining)

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, configs), nprocs=world_size)
    