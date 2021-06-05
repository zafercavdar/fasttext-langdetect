import os
from typing import Dict, Union

import fasttext
import wget

models = {"low_mem": None, "high_mem": None}


def download_model(name):
    url = f"https://dl.fbaipublicfiles.com/fasttext/supervised-models/{name}"
    target_folder = "/tmp/fasttext-langdetect"
    target_path = os.path.join(target_folder, name)
    if not os.path.exists(target_path):
        os.makedirs(target_folder, exist_ok=True)
        wget.download(url=url, out=target_path)
    return target_path


def get_or_load_model(low_memory=False):
    if low_memory:
        model = models.get("low_mem", None)
        if not model:
            model_path = download_model("lid.176.ftz")
            model = fasttext.load_model(model_path)
            models["low_mem"] = model
        return model
    else:
        model = models.get("high_mem", None)
        if not model:
            model_path = download_model("lid.176.bin")
            model = fasttext.load_model(model_path)
            models["high_mem"] = model
        return model


def detect(text: str, low_memory=False) -> Dict[str, Union[str, float]]:
    model = get_or_load_model(low_memory)
    labels, scores = model.predict(text)
    label = labels[0].replace("__label__", '')
    score = min(float(scores[0]), 1.0)
    return {
        "lang": label,
        "score": score,
    }
