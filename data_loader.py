import numpy as np
from datasets import MovingMNIST, MineRL, GQNMazes
from torch.utils.data import DataLoader
# import tensorflow.compat.v1 as tf
# import tensorflow_datasets as tfds
import torch




def load_dataset(cfg, **kwargs):
    if cfg.dataset == "minerl":
        train_data = MineRL(
            cfg.datadir,
            train=True,
        )
        test_data = MineRL(
            cfg.datadir,
            train=False,
        )
    elif cfg.dataset == "mmnist":
        train_data = MovingMNIST(
            cfg.datadir,
            train=True,
        )
        test_data = MovingMNIST(
            cfg.datadir,
            train=False,
        )
    elif cfg.dataset == "mazes":

        train_data_batch = GQNMazes(
            cfg.batch_size,
            cfg.num_epochs,
            train=True,
            seq_len=cfg.seq_len,
            data_root=cfg.datadir,
        ).get_batch()
        test_data_batch = GQNMazes(
            cfg.batch_size,
            1,
            train=False,
            seq_len=cfg.eval_seq_len,
            data_root=cfg.datadir,
        ).get_batch()

    else:
        raise ValueError("Dataset {} not supported.".format(cfg.dataset))
    
    if cfg.debug:
        train_data = torch.utils.data.Subset(train_data, range(200))
        test_data = torch.utils.data.Subset(test_data, range(10))

    # train_dataloader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=2)
    # test_dataloader = DataLoader(test_data, batch_size=8, shuffle=True)
    # return train_dataloader, test_dataloader
    return train_data, test_data


