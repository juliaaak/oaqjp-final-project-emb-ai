import requests

URL = "https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
HEADERS = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}
EMOTIONS = ("anger", "disgust", "fear", "joy", "sadness")


def _empty_result():
    result = {e: None for e in EMOTIONS}
    result["dominant_emotion"] = None
    return result


def emotion_detector(text_to_analyse):
    if not text_to_analyse or not text_to_analyse.strip():
        return _empty_result()

    payload = {"raw_document": {"text": text_to_analyse}}
    response = requests.post(URL, json=payload, headers=HEADERS, timeout=10)

    if response.status_code == 400:
        return _empty_result()
    response.raise_for_status()

    emotions = response.json()["emotionPredictions"][0]["emotion"]
    result = {e: emotions[e] for e in EMOTIONS}
    result["dominant_emotion"] = max(result, key=result.get)
    return result