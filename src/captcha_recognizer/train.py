import argparse
from pprint import pprint

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

from captcha_recognizer import utils
from captcha_recognizer.dataset import CaptchaDataloader
from captcha_recognizer.inference import CTCDecoder
from captcha_recognizer.models import CaptchaModel
from config import (
    BATCH_SIZE,
    DEVICE,
    DICT_PATH,
    EPOCHS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    LR,
    MODEL_PATH,
)


def train(model, no_epochs, train_loader, val_loader, lr, load_model=False):
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.8, patience=5, verbose=True
    )

    train_loss_all = []
    val_loss_all = []

    if not MODEL_PATH.exists():
        MODEL_PATH.mkdir()

    if load_model:
        utils.load_model(model, str(MODEL_PATH / "model.pt"))

    model.to(DEVICE)

    for epoch in range(no_epochs):
        epoch_loss = 0
        model.train()
        train_loader = tqdm(train_loader, total=len(train_loader))

        for image, target, _ in train_loader:
            optimizer.zero_grad()
            log_probs = model(image.to(DEVICE))
            log_probs = log_probs.permute(
                1, 0, 2
            )  # timestep, bs, values (input to CTC loss)
            input_lengths = torch.full(
                size=(log_probs.size(1),),
                fill_value=(log_probs.size(0)),
                dtype=torch.int32,
            )  # bs, 75
            target_lengths = torch.full(
                size=(target.size(0),), fill_value=(target.size(1)), dtype=torch.int32
            )  # bs, 5

            loss = nn.CTCLoss(blank=0)(
                log_probs,
                target.to(DEVICE),
                input_lengths.to(DEVICE),
                target_lengths.to(DEVICE),
            )

            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

        val_loss, val_acc = eval(model, val_loader)

        print(
            f"Epoch {epoch+1}: Train Loss = {epoch_loss/len(train_loader)}, Val Loss = {val_loss}, Val Acc = {val_acc}"
        )

        train_loss_all.append(epoch_loss / len(train_loader))
        val_loss_all.append(val_loss)

        scheduler.step(val_loss)

        # utils.save_model(model, str(MODEL_PATH/"model.pt"))

    plt.figure()
    plt.plot(train_loss_all, color="red", label="Train Loss")
    plt.plot(val_loss_all, color="blue", label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Train and Val Loss per Epoch")
    plt.legend()
    plt.savefig(str(MODEL_PATH / "Loss History.png"))


def eval(model, val_loader):

    model.eval()
    val_loss = 0
    correct_cnt = 0
    total_cnt = 0

    val_loader = tqdm(val_loader, total=len(val_loader))

    with torch.no_grad():
        for image, target, label in val_loader:
            log_probs = model(image.to(DEVICE))
            log_probs = log_probs.permute(1, 0, 2)
            input_lengths = torch.full(
                size=(log_probs.size(1),),
                fill_value=(log_probs.size(0)),
                dtype=torch.int32,
            )
            target_lengths = torch.full(
                size=(target.size(0),), fill_value=(target.size(1)), dtype=torch.int32
            )
            loss = nn.CTCLoss(blank=0)(
                log_probs,
                target.to(DEVICE),
                input_lengths.to(DEVICE),
                target_lengths.to(DEVICE),
            )

            val_loss += loss.item()

            decoder = CTCDecoder()

            _, pred_full, pred = decoder.decode(log_probs.permute(1, 0, 2))

            for i in range(len(label)):
                total_cnt += 1
                if label[i] == pred[i]:
                    correct_cnt += 1

        combined = list(zip(pred_full, pred, label))
        pprint(combined[:5])

    val_acc = correct_cnt / total_cnt

    return val_loss, val_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-bs", "--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("-lr", "--learning_rate", type=float, default=LR)
    parser.add_argument("-ep", "--epochs", type=int, default=EPOCHS)
    parser.add_argument("-l", "--load", type=bool, default=False)

    args = parser.parse_args()

    utils.seed_everything()
    data_loader = CaptchaDataloader(
        batch_size=args.batch_size, resize=(IMAGE_HEIGHT, IMAGE_WIDTH)
    )

    train_loader = data_loader.train_loader()
    val_loader = data_loader.val_loader()

    model = CaptchaModel(data_loader.full_dataset.index)

    train(model, args.epochs, train_loader, val_loader, args.learning_rate, args.load)

    # python3 src/captcha_recognizer/train.py -ep 2 -l False
