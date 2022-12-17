import glob
import os

import torch

from captcha_recognizer.dataset import CaptchaDataloader, CaptchaDataset
from config import DATA_PATH, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_WORKERS


def test_dataset():
    dataset = CaptchaDataset(image_paths=DATA_PATH, resize=(IMAGE_HEIGHT, IMAGE_WIDTH))
    dataset.get_targets()

    image_paths = glob.glob(os.path.join(DATA_PATH, "*.png"))

    # test size
    assert len(image_paths) == len(dataset), "Length dataset mismatched"
    assert len(dataset.id_char.keys()) == dataset.index, "Length label mismatched"

    # test sample
    image, target, img_label = dataset[4]
    sequence = "".join([dataset.id_char[idx] for idx in target.tolist()])
    assert sequence == dataset.image_paths[4].split("/")[-1][:-4], "Wrong label"
    assert image.size() == torch.Size(
        [3, IMAGE_HEIGHT, IMAGE_WIDTH]
    ), "Wrong image dimensions"
    assert len(img_label) == 5, "Wrong Label Length"


def test_loader():
    batch_size = 32
    val_split = 0.3
    h, w = 100, 300

    data_loader = CaptchaDataloader(
        data_dir=DATA_PATH,
        batch_size=batch_size,
        val_split=val_split,
        resize=(h, w),
        num_workers=NUM_WORKERS,
    )

    # test size
    assert len(data_loader.full_dataset) == 1040, "Wrong full dataset size"
    assert len(data_loader.train_set) == 728, "wrong train set size"
    assert len(data_loader.test_set) == 312, "wrong test set size"

    train_loader = data_loader.train_loader()
    test_loader = data_loader.val_loader()
    train_images, train_targets, train_labels = next(iter(train_loader))
    test_images, test_targets, test_labels = next(iter(test_loader))
    assert train_images.size() == torch.Size(
        [batch_size, 3, h, w]
    ), "Wrong train image dimension"
    assert test_images.size() == torch.Size(
        [batch_size, 3, h, w]
    ), "Wrong train image dimension"
    assert len(train_targets) == batch_size, "Wrong train batch size"
    assert len(test_targets) == batch_size, "Wrong test batch size"
    assert len(train_labels) == batch_size, "Wrong train label batch size"
    assert len(test_labels) == batch_size, "Wrong test label batch size"


if __name__ == "__main__":
    test_dataset()
    test_loader()
