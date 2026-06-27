from collections import deque
import numpy as np

# How each emotion maps to engagement level
# 1.0 = fully engaged, 0.0 = completely disengaged
EMOTION_ENGAGEMENT = {
    "happy":    1.0,   # enjoying the content
    "neutral":  0.7,   # paying attention, not excited
    "surprise": 0.8,   # something caught their attention
    "sad":      0.3,   # struggling or demotivated
    "fear":     0.2,   # stressed, possibly overwhelmed
    "angry":    0.1,   # frustrated
    "disgust":  0.1,   # strongly negative
    None:       0.0,   # no face detected = not at screen
}

ENGAGEMENT_LABELS = {
    (0.7, 1.0): "focused",
    (0.4, 0.7): "distracted",
    (0.0, 0.4): "disengaged",
}


class EngagementTracker:
    """
    Tracks a rolling window of emotion readings and
    produces a smoothed engagement score over time.

    This prevents a single bored frame from tanking
    the score — it looks at trends, not snapshots.
    """

    def __init__(self, window_size: int = 30):
        # Keep last N emotion readings (default = ~1 minute at 0.5 fps)
        self.window      = deque(maxlen=window_size)
        self.window_size = window_size

    def update(self, emotion_result: dict):
        """Feed in one result from detect_emotion_from_frame()."""
        emotion = emotion_result.get("emotion")
        score   = EMOTION_ENGAGEMENT.get(emotion, 0.0)

        # Weight by confidence — low confidence readings matter less
        confidence = emotion_result.get("confidence", 1.0)
        weighted   = score * confidence
        self.window.append(weighted)

    def get_engagement(self) -> dict:
        """
        Returns current engagement state based on rolling average.
        """
        if not self.window:
            return {"score": 0.0, "label": "no_data", "readings": 0}

        scores        = list(self.window)
        avg_score     = float(np.mean(scores))
        trend         = float(np.polyfit(range(len(scores)), scores, 1)[0]) if len(scores) > 1 else 0.0
        label         = self._label(avg_score)

        return {
            "score":    round(avg_score, 3),
            "label":    label,               # focused / distracted / disengaged
            "trend":    round(trend, 4),     # positive = getting more engaged
            "readings": len(scores),
        }

    def needs_intervention(self) -> bool:
        """
        Returns True if the student has been disengaged
        for more than half the tracking window.
        """
        if len(self.window) < self.window_size // 2:
            return False
        return np.mean(list(self.window)) < 0.3

    def reset(self):
        self.window.clear()

    @staticmethod
    def _label(score: float) -> str:
        for (low, high), label in ENGAGEMENT_LABELS.items():
            if low <= score <= high:
                return label
        return "disengaged"