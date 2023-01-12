import pathlib
import pickle

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from captcha_recognizer import models, utils
from captcha_recognizer.inference import CTCDecoder
from config import DEVICE, DICT_PATH, IMAGE_HEIGHT, IMAGE_WIDTH, MAIN_PATH, MODEL_PATH

model_path = str(MODEL_PATH / "model.pt")
dict_path = str(DICT_PATH / "decode_dict.pkl")
input = str(MAIN_PATH / "fastapi_backend" / "2b827.png")


class Predictor:
    def __init__(self, model_path, decoder_path):
        self.model_path = model_path
        self.decoder_path = decoder_path
        with open(self.decoder_path, "rb") as f:
            self.decode_dict = pickle.load(f)

        self.model = models.CaptchaModel(len(self.decode_dict) - 1)
        utils.load_model(self.model, self.model_path)

        self.model.eval()
        self.model.to(DEVICE)

        mean = [0.5, 0.5, 0.5]
        std = [0.2, 0.2, 0.2]

        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
            ]
        )

    def predict(self, input):
        image = Image.open(input).convert("RGB")

        image = np.array(image)
        image = self.transforms(image).unsqueeze(0)

        with torch.no_grad():
            log_probs = self.model(image.to(DEVICE))
            decoder = CTCDecoder()
            _, _, preds = decoder.decode(log_probs)

        return {"Prediction": preds[0]}


if __name__ == "__main__":
    predictor = Predictor(model_path, dict_path)
    output = predictor.predict(input)
    print(output)
