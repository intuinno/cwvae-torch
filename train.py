import numpy as np
import argparse
import os
import yaml
import torch
import pathlib
import sys

# from cwvae import build_model
# from loggers.summary import Summary
# from loggers.checkpoint import Checkpoint
from data_loader import *
import tools

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# def train_setup(cfg, loss):
#     session_config = tf.ConfigProto(device_count={"GPU": 1}, log_device_placement=False)
#     session = tf.Session(config=session_config)
#     step = tools.Step(session)

#     with tf.name_scope("optimizer"):
#         # Getting all trainable variables.
#         weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

#         # Creating optimizer.
#         optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.lr, epsilon=1e-04)

#         # Computing gradients.
#         grads = optimizer.get_gradients(loss, weights)
#         grad_norm = tf.global_norm(grads)

#         # Clipping gradients by global norm, and applying gradient.
#         if cfg.clip_grad_norm_by is not None:
#             capped_grads = tf.clip_by_global_norm(grads, cfg.clip_grad_norm_by)[0]
#             capped_gvs = [
#                 tuple((capped_grads[i], weights[i])) for i in range(len(weights))
#             ]
#             apply_grads = optimizer.apply_gradients(capped_gvs)
#         else:
#             gvs = zip(grads, weights)
#             apply_grads = optimizer.apply_gradients(gvs)
#     return apply_grads, grad_norm, session, step


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs='+', required=True)
    args, remaining = parser.parse_known_args()
    configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0].parent / 'configs.yml').read_text())
    )
    defaults = {}
    for name in args.configs:
        defaults.update(configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f'--{key}', type=arg_type, default=arg_type(value))
    configs = parser.parse_args(remaining)

    
    # Creating model dir with experiment name.
    exp_rootdir = os.path.join(cfg.logdir, cfg.dataset, tools.exp_name(cfg))
    os.makedirs(exp_rootdir, exist_ok=True)

    # Dumping config.
    print(cfg)
    with open(os.path.join(exp_rootdir, "config.yml"), "w") as f:
        yaml.dump(dict(cfg), f, default_flow_style=False)

    # Load dataset.
    train_dataloader, val_dataloader = load_dataset(cfg)

    for epoch in range(cfg.num_epoch):
        
        for i, x in enumerate(train_dataloader):
            x = x.to(device)

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
