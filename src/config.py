from pathlib import Path

import torch

MAIN_PATH = Path(__file__).resolve().parents[1]
DATA_PATH = MAIN_PATH / "data" / "captcha_images_v2"
MODEL_PATH = MAIN_PATH / "artifact" / "model_checkpoint"
DICT_PATH = MAIN_PATH / "artifact"

BATCH_SIZE = 32
IMAGE_WIDTH = 300
IMAGE_HEIGHT = 75
NUM_WORKERS = 8
EPOCHS = 300
LR = 1e-3
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
