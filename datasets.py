import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from torch import Tensor

from torchvision.datasets.folder import find_classes, make_dataset
from torchvision.datasets.video_utils import VideoClips
from torchvision.datasets.vision import VisionDataset

from pathlib import Path
from PIL import Image
import os
import os.path
import errno
import numpy as np
import torch
import codecs
from zipfile import ZipFile
import urllib


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

    def __init__(self, root, train=True, transform=None, download=True):
        self.root = Path(root).expanduser() / self.data_folder
        self.transform = transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')
        
        super().__init__(self.root)
        
        print("Processing")

        self.split_folder = self.root / self.raw_folder
            
        self.samples = make_dataset(self.split_folder, extensions='.mp4')
        
        if self.train:
            label = 0
        else:
            label = 1
            
        video_list = [x[0] for x in self.samples if x[1] == label]
        self.video_clips = VideoClips(
            video_list,
            num_workers = 4
        )

    def __getitem__(self, idx: int):
        video, audio, info, video_idx = self.video_clips.get_clip(idx)

        video = video.to(torch.float32) / 255.0

        if self.transform is not None:
            video = self.transform(video)

        return video


    def _check_exists(self):
        return os.path.exists( self.root / self.raw_folder / "test-seq1000") and \
            os.path.exists(self.root / self.raw_folder / "train-seq100")

    def download(self):
        """Download the Moving MNIST data if it doesn't exist in processed_folder already."""

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        url = self.data_url
        print('Downloading ' + url)
        data = urllib.request.urlopen(url)
        filename = url.rpartition('/')[-1]
        file_path = self.root / self.raw_folder / filename
        with open(file_path, 'wb') as f:
            f.write(data.read())
        extract_path = self.root / self.raw_folder
        with ZipFile(file_path, 'r') as zObject:
            zObject.extractall(path=extract_path)



if __name__ == "__main__":
    a = MovingMNIST("datasets", train=True, download=True)
    b = a.__getitem__(0)
    print(b)