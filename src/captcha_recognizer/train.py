import argparse

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

from captcha_recognizer import utils
from captcha_recognizer.dataset import CaptchaDataloader
from captcha_recognizer.models import CaptchaModel
from config import DEVICE, DICT_PATH, EPOCHS, MODEL_PATH


def train(model, no_epochs, train_loader, val_loader, lr, load_model=False):
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.8, patience=5, verbose=True
    )

    train_loss = []
    val_loss = []

    if MODEL_PATH.exists():
        MODEL_PATH.mkdir()

    if load_model:
        utils.load_model(model, str(MODEL_PATH / "model.pt"))

    model.to(DEVICE)

    for epoch in range(no_epochs):
        pass
