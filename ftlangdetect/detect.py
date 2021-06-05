from typing import Dict, Union

import fasttext

models = {"low_mem": None, "high_mem": None}


def get_or_load_model(low_memory=False):
    if low_memory:
        model = models.get("low_mem", None)
        if not model:
            model = fasttext.load_model("ftlangdetect/models/lid.176.ftz")
            models["low_mem"] = model
        return model
    else:
        model = models.get("high_mem", None)
        if not model:
            model = fasttext.load_model("ftlangdetect/models/lid.176.bin")
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
