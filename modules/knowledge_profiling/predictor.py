import joblib
import pandas as pd
from config.settings import MODEL_DIR, TOPICS
from modules.knowledge_profiling.feature_engineering import (
    build_student_features,
    get_weak_topics,
)

_model         = joblib.load(MODEL_DIR / "knowledge_model.pkl")
_scaler        = joblib.load(MODEL_DIR / "scaler.pkl")
_feature_names = joblib.load(MODEL_DIR / "feature_names.pkl")

PERF_LABELS = {0: "weak", 1: "average", 2: "strong"}


def predict(quiz_history: list) -> dict:
    df       = pd.DataFrame(quiz_history)
    features = build_student_features(df)
    X        = features[_feature_names]
    X_sc     = _scaler.transform(X)

    label_id   = int(_model.predict(X_sc)[0])
    confidence = float(_model.predict_proba(X_sc)[0][label_id])
    row        = features.iloc[0]
    weak       = get_weak_topics(row)

    return {
        "performance":    PERF_LABELS[label_id],
        "confidence":     round(confidence, 3),
        "avg_score":      round(float(row["avg_score"]), 3),
        "score_trend":    round(float(row["score_trend"]), 4),
        "weak_topics":    weak,
        "topic_scores":   {t: round(float(row.get(f"{t}_score", 0)), 2) for t in TOPICS},
        "recommendation": _recommend(label_id, weak, float(row["score_trend"])),
    }


def _recommend(label: int, weak: list, trend: float) -> str:
    if label == 0:
        if trend > 0.01:
            return f"You're improving! Keep focusing on: {', '.join(weak[:3])}."
        return f"Start with basics in: {', '.join(weak[:3])}. Short daily sessions work best."
    elif label == 1:
        if weak:
            return f"Good work! Strengthen these areas next: {', '.join(weak[:2])}."
        return "Consistent performance. Try harder questions to level up."
    return "Excellent! Try advanced problems and consider peer teaching."