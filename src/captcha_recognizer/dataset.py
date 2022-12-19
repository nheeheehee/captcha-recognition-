import glob
import os
import pickle
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

from config import DATA_PATH, DICT_PATH, NUM_WORKERS

ImageFile.LOAD_TRUNCATED_IMAGES = True


class CaptchaDataset(Dataset):
    """Generate tensors as input and labels as output for Captcha images

    Args:
        Dataset: Captcha images

    """

    def __init__(
        self, 
        image_paths: Union[Path, str] 
        = DATA_PATH, 
        resize: Optional[tuple] 
        = None
    ):
        """Defines dataset and transformation steps

        Args:
            image_paths (Union[Path, str], optional): directory to the png images.
                                                        Defaults to DATA_PATH.
            resize (Optional[tuple], optional): a tuple of (height, width) that 
                                                the image should be resized to. 
                                                Defaults to None.
        """

        self.image_paths = glob.glob(os.path.join(image_paths, "*.png"))

        mean = [0.5, 0.5, 0.5]
        std = [0.2, 0.2, 0.2]

        if resize:
            self.transforms = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                    transforms.Resize(resize),
                ]
            )
        else:
            self.transforms = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean, std)]
            )

    def get_targets(self):
        """Get numeric labels for captcha alphanumeric sequences"""
        self.target_sequence = [image.split("/")[-1][:-4] \
                                for image in self.image_paths]
        self.index = 0
        self.char_id = {}
        self.id_char = {}
        for sequence in self.target_sequence:
            for char in sequence:
                if char not in self.char_id:
                    self.index += 1
                    self.char_id[char] = self.index
                    self.id_char[self.index] = char

    def save_target_dict(self, path: str):
        """save the matching dict of numeric labels to alphanumeric character of captcha sequence

        Args:
            path (str): path to save the matching dict
        """
        file = open(path, "wb")
        pickle.dump(self.id_char, file)
        file.close()

    def load_dict(self, path: str) -> dict:
        """Load decoding dictionary

        Args:
            path (str): path that decoding dict was saved

        Returns:
            dict: decode dict for
        """
        file = open(path, "rb")
        decode_dict = pickle.load(file)
        file.close()
        return decode_dict

    def __len__(self):
        """Magic method to return the number of images in the directory"""
        return len(self.image_paths)

    def __getitem__(self, item: int) -> tuple:
        """Magic method to get a sample of input and output

        Args:
            item (int): an index within the range of the size of the datast

        Returns:
            tuple: original image array, tensor of the transformed image array, tensor of label array
        """

        image = Image.open(self.image_paths[item]).convert(
            "RGB"
        )  # convert 4 channel image to 3 channels
        target = [self.char_id[char] for char in self.target_sequence[item]]
        image = np.array(image)
        image = self.transforms(image)

        return image, torch.tensor(target, dtype=torch.long), self.target_sequence[item]


class CaptchaDataloader:
    """Generate Train and Test dataloader for Dataset"""

    def __init__(
        self,
        data_dir: Union[Path, str] = DATA_PATH,
        batch_size: int = 32,
        val_split: float = 0.2,
        resize: Optional[Sequence] = None,
        num_workers=NUM_WORKERS,
    ):

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.full_dataset = CaptchaDataset(data_dir, resize=resize)

        self.full_dataset.get_targets()
        self.full_dataset.save_target_dict(str(DICT_PATH / "decode_dict.pkl"))

        if val_split > 0:
            full_size = len(self.full_dataset)
            test_size = int(np.floor(full_size * val_split))
            train_size = full_size - test_size
            self.train_set, self.test_set = random_split(
                self.full_dataset, [train_size, test_size]
            )
        else:
            self.train_set = self.full_dataset
            self.test_set = None

    def train_loader(self):
        """generate train dataloader"""
        return DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=True, drop_last=True
        )

    def val_loader(self):
        """generate test dataloader
        Raises:
            Exception: if no validation split was set up
        """
        if self.test_set:
            return DataLoader(
                self.test_set,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False,
            )

        else:
            raise Exception("Pls set up validation split")


if __name__ == "__main__":
    dataset = CaptchaDataset()
    dataset.get_targets()
    print(dataset[0])

    dataloader = CaptchaDataloader(batch_size=10)
    trainloader = dataloader.train_loader()
    sample_batch = next(iter(trainloader))
    print(sample_batch)
