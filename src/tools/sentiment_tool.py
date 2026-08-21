from __future__ import annotations

from typing import Any
from langchain_core.tools import tool
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()


@tool
def sentiment_classification_tool(text: str) -> dict[str, Any]:
    """
    Classify customer-support sentiment with VADER.

    Returns label, compound score, component scores, and the model name.
    Uses VADER's standard compound thresholds:
      >= 0.05 positive
      <= -0.05 negative
      otherwise neutral
    """
    if not text or not text.strip():
        return {
            "label": "neutral",
            "compound": 0.0,
            "scores": {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0},
            "model": "VADER",
        }

    scores = _analyzer.polarity_scores(text)
    compound = float(scores["compound"])

    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"

    return {
        "label": label,
        "compound": compound,
        "scores": {k: float(v) for k, v in scores.items()},
        "model": "VADER",
    }
