import numpy as np
from collections import deque


class DifficultyAdjuster:
    """
    Adaptive difficulty engine that adjusts question difficulty
    in real time based on student performance.

    Uses a simple but effective algorithm:
    - Tracks rolling accuracy over last N answers
    - Moves difficulty up if student is doing too well
    - Moves difficulty down if student is struggling
    - Keeps difficulty stable when performance is in the target zone
    """

    LEVELS = ["very_easy", "easy", "medium", "hard", "very_hard"]

    # Target accuracy range — if student is in this zone, difficulty stays
    TARGET_LOW  = 0.60   # below this → make easier
    TARGET_HIGH = 0.85   # above this → make harder

    def __init__(self, starting_level: str = "medium", window_size: int = 5):
        if starting_level not in self.LEVELS:
            starting_level = "medium"

        self.current_level   = starting_level
        self.window          = deque(maxlen=window_size)
        self.history         = []   # full record of every answer
        self.questions_asked = 0
        self.correct_total   = 0

    def record_answer(self, is_correct: bool, time_taken_sec: float = None) -> dict:
        """
        Record a student's answer and adjust difficulty if needed.

        Args:
            is_correct:      whether the student got the question right
            time_taken_sec:  how long they took (optional, used for insight)

        Returns:
            dict with new difficulty level and feedback message
        """
        self.window.append(1 if is_correct else 0)
        self.questions_asked += 1
        if is_correct:
            self.correct_total += 1

        self.history.append({
            "question_num":  self.questions_asked,
            "is_correct":    is_correct,
            "difficulty":    self.current_level,
            "time_taken":    time_taken_sec,
        })

        old_level = self.current_level
        adjustment = self._compute_adjustment()
        message    = self._get_feedback(is_correct, adjustment)

        return {
            "previous_level": old_level,
            "current_level":  self.current_level,
            "adjusted":       old_level != self.current_level,
            "rolling_accuracy": self.get_rolling_accuracy(),
            "overall_accuracy": self.get_overall_accuracy(),
            "feedback":       message,
        }

    def _compute_adjustment(self) -> str:
        """Decide whether to go up, down, or stay based on rolling accuracy."""
        if len(self.window) < 3:
            return "stable"   # not enough data yet

        accuracy = np.mean(list(self.window))
        idx      = self.LEVELS.index(self.current_level)

        if accuracy > self.TARGET_HIGH and idx < len(self.LEVELS) - 1:
            self.current_level = self.LEVELS[idx + 1]
            return "harder"
        elif accuracy < self.TARGET_LOW and idx > 0:
            self.current_level = self.LEVELS[idx - 1]
            return "easier"

        return "stable"

    def _get_feedback(self, is_correct: bool, adjustment: str) -> str:
        messages = {
            ("correct", "harder"):  "Excellent! You're ready for harder questions.",
            ("correct", "stable"):  "Great job! Keep it up.",
            ("correct", "easier"):  "Good answer! Let's try a slightly harder one.",
            ("wrong",   "easier"):  "No worries! Let's try something a bit simpler.",
            ("wrong",   "stable"):  "Not quite. Review this concept and try again.",
            ("wrong",   "harder"):  "Keep going — you're improving overall.",
        }
        key = ("correct" if is_correct else "wrong", adjustment)
        return messages.get(key, "Keep going!")

    def get_rolling_accuracy(self) -> float:
        if not self.window:
            return 0.0
        return round(float(np.mean(list(self.window))), 3)

    def get_overall_accuracy(self) -> float:
        if self.questions_asked == 0:
            return 0.0
        return round(self.correct_total / self.questions_asked, 3)

    def get_session_summary(self) -> dict:
        return {
            "total_questions":   self.questions_asked,
            "correct_answers":   self.correct_total,
            "overall_accuracy":  self.get_overall_accuracy(),
            "final_difficulty":  self.current_level,
            "difficulty_history": [h["difficulty"] for h in self.history],
        }

    def recommended_next_difficulty(self) -> str:
        return self.current_level