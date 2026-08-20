from transformers import pipeline


emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None
)


def emotion_detector(text_to_analyze):
    """Detect emotions in English text using a local transformer model."""

    if not text_to_analyze or not text_to_analyze.strip():
        return {
            "anger": None,
            "disgust": None,
            "fear": None,
            "joy": None,
            "sadness": None,
            "dominant_emotion": None
        }

    results = emotion_classifier(text_to_analyze)

    if results and isinstance(results[0], list):
        results = results[0]

    emotion_scores = {
        result["label"]: result["score"]
        for result in results
    }

    five_emotions = {
        "anger": emotion_scores["anger"],
        "disgust": emotion_scores["disgust"],
        "fear": emotion_scores["fear"],
        "joy": emotion_scores["joy"],
        "sadness": emotion_scores["sadness"]
    }

    dominant_emotion = max(
        five_emotions,
        key=five_emotions.get
    )

    return {
        "anger": five_emotions["anger"],
        "disgust": five_emotions["disgust"],
        "fear": five_emotions["fear"],
        "joy": five_emotions["joy"],
        "sadness": five_emotions["sadness"],
        "dominant_emotion": dominant_emotion
    }