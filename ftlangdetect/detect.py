from typing import Dict, Union
import fasttext

ft_model = fasttext.load_model("ftlangdetect/model/lid.176.bin")


def detect(text: str) -> Dict[str, Union[str, float]]:
    labels, scores = ft_model.predict(text)
    label = labels[0].replace("__label__", '')
    score = min(float(scores[0]), 1.0)
    return {
        "lang": label,
        "score": score,
    }
