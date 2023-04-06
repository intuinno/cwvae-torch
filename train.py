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


# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using {device} device")




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
    
    now = datetime.now()
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
            
        # Write evaluation summary
        print(f'======== Epoch {epoch} / {configs.num_epochs} ==========')
        now = datetime.now()
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
            post, context, met = model.train(x)
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
            
          
            
        

    # Build model.
    model_components = build_model(cfg)
    model = model_components["meta"]["model"]

    # Setting up training.
    apply_grads, grad_norm, session, step = train_setup(cfg, model.loss)

    # Define summaries.
    summary = Summary(exp_rootdir, save_gifs=cfg.save_gifs)
    summary.build_summary(cfg, model_components, grad_norm=grad_norm)

    # Define checkpoint saver for variables currently in session.
    checkpoint = Checkpoint(exp_rootdir)

    # Restore model (if exists).
    if os.path.exists(checkpoint.log_dir_model):
        print("Restoring model from {}".format(checkpoint.log_dir_model))
        checkpoint.restore(session)
        print("Will start training from step {}".format(step()))
    else:
        # Initialize all variables.
        session.run(tf.global_variables_initializer())

    # Start training.
    print("Getting validation batches.")
    val_batches = get_multiple_batches(val_data_batch, cfg.num_val_batches, session)
    print("Training.")
    while True:
        try:
            train_batch = get_single_batch(train_data_batch, session)
            feed_dict_train = {model_components["training"]["obs"]: train_batch}
            feed_dict_val = {model_components["training"]["obs"]: val_batches}

            # Train one step.
            session.run(fetches=apply_grads, feed_dict=feed_dict_train)

            # Saving scalar summaries.
            if step() % cfg.save_scalars_every == 0:
                summaries = session.run(
                    summary.scalar_summary, feed_dict=feed_dict_train
                )
                summary.save(summaries, step(), True)
                summaries = session.run(summary.scalar_summary, feed_dict=feed_dict_val)
                summary.save(summaries, step(), False)

            # Saving gif summaries.
            if step() % cfg.save_gifs_every == 0:
                summaries = session.run(summary.gif_summary, feed_dict=feed_dict_train)
                summary.save(summaries, step(), True)
                summaries = session.run(summary.gif_summary, feed_dict=feed_dict_val)
                summary.save(summaries, step(), False)

            # Saving model.
            if step() % cfg.save_model_every == 0:
                checkpoint.save(session)

            if cfg.save_named_model_every and step() % cfg.save_named_model_every == 0:
                checkpoint.save(session, save_dir="model_{}".format(step()))

            step.increment()
        except tf.errors.OutOfRangeError:
            break

    print("Training complete.")
