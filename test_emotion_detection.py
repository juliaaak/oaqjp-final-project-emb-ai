from EmotionDetection.emotion_detection import emotion_detector


def test_joy():
    assert emotion_detector("I am glad this happened")["dominant_emotion"] == "joy"

def test_anger():
    assert emotion_detector("I am really mad about this")["dominant_emotion"] == "anger"

def test_empty_input():
    assert emotion_detector("")["dominant_emotion"] is None