import pathlib

from fastapi import APIRouter, Depends, File, UploadFile

from config import DICT_PATH, MODEL_PATH
from fastapi_backend.captcha_predictor import Predictor

model_path = str(MODEL_PATH / "model.pt")
dict_path = str(DICT_PATH / "decode_dict.pkl")

predictor = Predictor(model_path, dict_path)

router = APIRouter(prefix="/captcha")


@router.get("/")
def service_type():
    return {"Service": "Translate Captcha Images to Text"}


@router.post("/predict")
def predict(file: UploadFile = File(...)):
    try:
        output = predictor.predict(file.file)
        return output

    except Exception:
        return {"Error": "Error Uploading File"}

    finally:
        file.file.close()
