import argparse
import torch
import pathlib

from cwvae import CWVAE
from data_loader import *
from datetime import datetime
import pytz
import ruamel.yaml as yaml

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logdir",
        default=None, 
        type=str,
        required=True,
        help='path to dir containing model and config.yml'
    )
    parser.add_argument(
        "--use-obs",
        default=None,
        type=str,
        help="string of T/Fs per level, e.g. TTF to skip obs at the top level"
    )

    args = parser.parse_args()
    
    exp_rootdir = pathlib.Path(args.logdir).expanduser()
    assert exp_rootdir.is_dir()
    print(f"Reading log dir {exp_rootdir}")
    
    tz = pytz.timezone("US/Central")
    now = datetime.now(tz)
    date_time = now.strftime("%Y%m%d_%H%M%S")
    eval_logdir = exp_rootdir / f"eval_{datetime}"
    eval_logdir.mkdir()
    print(f"Saving eval results at {eval_logdir}")
    configs = yaml.safe_load((exp_rootdir / 'config.yml').read_text())
    configs.batch_size = 1 
    
    if args.use_obs is not None:
        assert len(args.use_obs) == configs.levels
        configs.use_obs = [dict(T=True, F=False)[c] for c in args.use_obs.upper()]
    else:
        configs.use_obs = True
    
    _, val_dataloader = load_dataset(configs)

    model = CWVAE(configs).to(configs.device)
    
