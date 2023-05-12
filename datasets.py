
from typing import Any, Callable, Dict, List, Optional, Tuple

from torch import Tensor

from torchvision.datasets.folder import find_classes, make_dataset
from torchvision.datasets.video_utils import VideoClips
from torchvision.datasets.vision import VisionDataset
import torchvision.transforms.v2 as transforms

from pathlib import Path
from PIL import Image

import errno
import numpy as np
import torch
import codecs
from zipfile import ZipFile
import urllib
import einops

class GQNMazes(VisionDataset):
    """
    `GQNMaze dataset for Clockwork VAE  <https://danijar.com/project/cwvae/>`_ dataset.

    # GQNMazes dataset

References:
```
@article{saxena2021clockworkvae,
  title={Clockwork Variational Autoencoders}, 
  author={Saxena, Vaibhav and Ba, Jimmy and Hafner, Danijar},
  journal={arXiv preprint arXiv:2102.09532},
  year={2021},
}
```
```

```
    Args:
        root (string): Root directory of dataset where ``moving_mnist/processed/training.pt``
            and  ``moving_mnist/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        split (int, optional): Train/test split size. Number defines how many samples
            belong to test set. 
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in ``root/moving_mnist/downloaded`` directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in an PIL
            image and returns a transformed version. E.g, ``transforms.RandomCrop``
    Args:
        root (string): Root directory of the Dataset.
        frames_per_clip (int): number of frames in a clip.
        train (bool, optional): if ``True``, creates a dataset from the train split,
            otherwise from the ``test`` split.
        transform (callable, optional): A function/transform that  takes in a TxHxWxC video
            and returns a transformed version.
        output_format (str, optional): The format of the output video tensors (before transforms).
            Can be either "THWC" (default) or "TCHW".

    Returns:
        tuple: A dict with the following entries:

            - video (Tensor[T, H, W, C] or Tensor[T, C, H, W]): The `T` video frames
    """

    data_url = "https://archive.org/download/gqn_mazes/gqn_mazes.zip"
    data_folder = "gqn_mazes"
    raw_folder = 'downloaded'
    processed_folder = 'processed'
    training_file = 'maze_train.pt'
    test_file = 'maze_test.pt'

    def __init__(self, root, train=True, transform=None, download=False, output_format="THWC"):
        self.root = Path(root).expanduser() / self.data_folder
        self.transform = transform
        self.train = train  # training set or test set
        data_dir = self.root / self.processed_folder

        if download or not self._check_exists():
            self.download()
        
        super().__init__(self.root)

        if self.train:
            label = 1
            save_filename = self.training_file
            clip_length = 100
        else:
            label = 0
            save_filename = self.test_file
            clip_length = 500
        
        
        if self._check_exists():
            self.video_clips = torch.load(data_dir / save_filename)
            
        else:        
            print("Processing")
            self.split_folder = self.root / self.raw_folder 
            self.samples = make_dataset(self.split_folder, extensions='.mp4')
            data_dir.mkdir(parents=True, exist_ok=True)
            video_list = [x[0] for x in self.samples if x[1] == label]
            self.video_clips = VideoClips(
                video_list,
                num_workers=4,
                output_format=output_format,
                clip_length_in_frames=clip_length
            )
            torch.save(self.video_clips, data_dir / save_filename )


            
    def __len__(self) -> int:
        return self.video_clips.num_clips()

    def __getitem__(self, idx: int):
        video, audio, info, video_idx = self.video_clips.get_clip(idx)
        
        video = einops.rearrange(video, 't h w c -> t c h w')
        # video = self.gray_scaler(video)
        video = einops.rearrange(video, 't c h w -> t h w c')

        # video = video.to(torch.float32) / 255.0

        if self.transform is not None:
            video = self.transform(video)

        return video


    def _check_exists(self):
        test_file = self.root / self.processed_folder / self.test_file
        train_file = self.root / self.processed_folder / self.training_file
        return test_file.is_file() and train_file.is_file()

    def download(self):
        """Download the Moving MNIST data if it doesn't exist in processed_folder already."""

        dir = self.root / self.raw_folder
        url = self.data_url
        filename = url.rpartition('/')[-1]
        file_path = self.root / self.raw_folder / filename

        if file_path.is_file():
            return 
        
        dir.mkdir(parents=True, exist_ok=True)
        
        print('Downloading ' + url)
        data = urllib.request.urlopen(url)
        
        with open(file_path, 'wb') as f:
            f.write(data.read())
        extract_path = self.root / self.raw_folder
        with ZipFile(file_path, 'r') as zObject:
            zObject.extractall(path=extract_path)

class MineRL(VisionDataset):
    """
    `MineRL dataset for Clockwork VAE  <https://danijar.com/project/cwvae/>`_ dataset.

    # MineRL dataset

References:
```
@article{saxena2021clockworkvae,
  title={Clockwork Variational Autoencoders}, 
  author={Saxena, Vaibhav and Ba, Jimmy and Hafner, Danijar},
  journal={arXiv preprint arXiv:2102.09532},
  year={2021},
}
```
```

```
    Args:
        root (string): Root directory of dataset where ``moving_mnist/processed/training.pt``
            and  ``moving_mnist/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        split (int, optional): Train/test split size. Number defines how many samples
            belong to test set. 
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in ``root/moving_mnist/downloaded`` directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in an PIL
            image and returns a transformed version. E.g, ``transforms.RandomCrop``
    Args:
        root (string): Root directory of the Dataset.
        frames_per_clip (int): number of frames in a clip.
        train (bool, optional): if ``True``, creates a dataset from the train split,
            otherwise from the ``test`` split.
        transform (callable, optional): A function/transform that  takes in a TxHxWxC video
            and returns a transformed version.
        output_format (str, optional): The format of the output video tensors (before transforms).
            Can be either "THWC" (default) or "TCHW".

    Returns:
        tuple: A dict with the following entries:

            - video (Tensor[T, H, W, C] or Tensor[T, C, H, W]): The `T` video frames
    """

    data_url = "https://archive.org/download/minerl_navigate/minerl_navigate.zip"
    data_folder = "mineRL"
    raw_folder = 'downloaded'
    processed_folder = 'processed'
    training_file = 'mineRL_train.pt'
    test_file = 'mineRL_test.pt'

    def __init__(self, root, train=True, transform=None, download=False, output_format="THWC"):
        self.root = Path(root).expanduser() / self.data_folder
        self.transform = transform
        self.train = train  # training set or test set
        data_dir = self.root / self.processed_folder

        if download or not self._check_exists():
            self.download()
        
        super().__init__(self.root)

        if self.train:
            label = 1
            save_filename = self.training_file
            clip_length = 100
        else:
            label = 0
            save_filename = self.test_file
            clip_length = 500
        
        
        if self._check_exists():
            self.video_clips = torch.load(data_dir / save_filename)
            
        else:        
            print("Processing")
            self.split_folder = self.root / self.raw_folder / 'minerl_navigate'
            self.samples = make_dataset(self.split_folder, extensions='.mp4')
            data_dir.mkdir(parents=True, exist_ok=True)
            video_list = [x[0] for x in self.samples if x[1] == label]
            self.video_clips = VideoClips(
                video_list,
                num_workers=4,
                output_format=output_format,
                clip_length_in_frames=clip_length
            )
            torch.save(self.video_clips, data_dir / save_filename )


            
    def __len__(self) -> int:
        return self.video_clips.num_clips()

    def __getitem__(self, idx: int):
        video, audio, info, video_idx = self.video_clips.get_clip(idx)
        
        video = einops.rearrange(video, 't h w c -> t c h w')
        # video = self.gray_scaler(video)
        video = einops.rearrange(video, 't c h w -> t h w c')

        # video = video.to(torch.float32) / 255.0

        if self.transform is not None:
            video = self.transform(video)

        return video


    def _check_exists(self):
        test_file = self.root / self.processed_folder / self.test_file
        train_file = self.root / self.processed_folder / self.training_file
        return test_file.is_file() and train_file.is_file()

    def download(self):
        """Download the Moving MNIST data if it doesn't exist in processed_folder already."""

        dir = self.root / self.raw_folder
        url = self.data_url
        filename = url.rpartition('/')[-1]
        file_path = self.root / self.raw_folder / filename

        if file_path.is_file():
            return 
        
        dir.mkdir(parents=True, exist_ok=True)
        
        print('Downloading ' + url)
        data = urllib.request.urlopen(url)
        
        with open(file_path, 'wb') as f:
            f.write(data.read())
        extract_path = self.root / self.raw_folder
        with ZipFile(file_path, 'r') as zObject:
            zObject.extractall(path=extract_path)



class MovingMNIST(VisionDataset):
    """
    `Moving MNIST for Clockwork VAE  <https://danijar.com/project/cwvae/>`_ dataset.

    # Moving MNIST Dataset

References:
```
@article{saxena2021clockworkvae,
  title={Clockwork Variational Autoencoders}, 
  author={Saxena, Vaibhav and Ba, Jimmy and Hafner, Danijar},
  journal={arXiv preprint arXiv:2102.09532},
  year={2021},
}
```
```

```
    Args:
        root (string): Root directory of dataset where ``moving_mnist/processed/training.pt``
            and  ``moving_mnist/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        split (int, optional): Train/test split size. Number defines how many samples
            belong to test set. 
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in ``root/moving_mnist/downloaded`` directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in an PIL
            image and returns a transformed version. E.g, ``transforms.RandomCrop``
    Args:
        root (string): Root directory of the Dataset.
        frames_per_clip (int): number of frames in a clip.
        train (bool, optional): if ``True``, creates a dataset from the train split,
            otherwise from the ``test`` split.
        transform (callable, optional): A function/transform that  takes in a TxHxWxC video
            and returns a transformed version.
        output_format (str, optional): The format of the output video tensors (before transforms).
            Can be either "THWC" (default) or "TCHW".

    Returns:
        tuple: A dict with the following entries:

            - video (Tensor[T, H, W, C] or Tensor[T, C, H, W]): The `T` video frames
    """

    data_url = "https://archive.org/download/moving_mnist/moving_mnist_2digit.zip"
    data_folder = "moving_mnist"
    raw_folder = 'downloaded'
    processed_folder = 'processed'
    training_file = 'moving_mnist_train.pt'
    test_file = 'moving_mnist_test.pt'

    def __init__(self, root, train=True, transform=None, download=False, output_format="THWC"):
        self.root = Path(root).expanduser() / self.data_folder
        self.transform = transform
        self.train = train  # training set or test set
        data_dir = self.root / self.processed_folder

        if download or not self._check_exists():
            self.download()
        
        super().__init__(self.root)

        if self.train:
            label = 1
            save_filename = self.training_file
            clip_length = 100
        else:
            label = 0
            save_filename = self.test_file
            clip_length = 1000
        
        
        if self._check_exists():
            self.video_clips = torch.load(data_dir / save_filename)
            
        else:        
            print("Processing")
            self.split_folder = self.root / self.raw_folder
            self.samples = make_dataset(self.split_folder, extensions='.mp4')
            data_dir.mkdir(parents=True, exist_ok=True)
            video_list = [x[0] for x in self.samples if x[1] == label]
            self.video_clips = VideoClips(
                video_list,
                num_workers=4,
                output_format=output_format,
                clip_length_in_frames=clip_length,
                # _video_width=32,
                # _video_height=32
            )
            torch.save(self.video_clips, data_dir / save_filename )

        self.gray_scaler = transforms.Grayscale()


            
    def __len__(self) -> int:
        return self.video_clips.num_clips()

    def __getitem__(self, idx: int):
        video, audio, info, video_idx = self.video_clips.get_clip(idx)
        
        video = einops.rearrange(video, 't h w c -> t c h w')
        video = self.gray_scaler(video)
        video = einops.rearrange(video, 't c h w -> t h w c')

        video = video.to(torch.float32) / 255.0

        if self.transform is not None:
            video = self.transform(video)

        return video


    def _check_exists(self):
        test_file = self.root / self.processed_folder / self.test_file
        train_file = self.root / self.processed_folder / self.training_file
        return test_file.is_file() and train_file.is_file()

    def download(self):
        """Download the Moving MNIST data if it doesn't exist in processed_folder already."""

        dir = self.root / self.raw_folder
        url = self.data_url
        filename = url.rpartition('/')[-1]
        file_path = self.root / self.raw_folder / filename

        if file_path.is_file():
            return 
        
        dir.mkdir(parents=True, exist_ok=True)
        
        print('Downloading ' + url)
        data = urllib.request.urlopen(url)
        
        with open(file_path, 'wb') as f:
            f.write(data.read())
        extract_path = self.root / self.raw_folder
        with ZipFile(file_path, 'r') as zObject:
            zObject.extractall(path=extract_path)



if __name__ == "__main__":
    a = MovingMNIST("datasets", train=True, download=True)
    b = a.__getitem__(0)
    print(b)