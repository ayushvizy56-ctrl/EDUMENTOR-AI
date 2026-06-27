import numpy as np
import pandas as pd
from config.settings import TOPICS, DATA_PROC


def build_student_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for student_id, attempts in df.groupby("student_id"):
        row = {"student_id": student_id}

        row["avg_score"]      = attempts["score"].mean()
        row["score_std"]      = attempts["score"].std()
        row["avg_time"]       = attempts["time_spent_sec"].mean()
        row["total_hints"]    = attempts["hints_used"].sum()
        row["hint_rate"]      = attempts["hints_used"].mean()
        row["total_attempts"] = len(attempts)
        row["avg_difficulty"] = attempts["difficulty"].mean()
        row["score_trend"]    = _slope(
            attempts["quiz_number"].values,
            attempts["score"].values
        )

        for topic in TOPICS:
            t = attempts[attempts["topic"] == topic]
            if t.empty:
                row[f"{topic}_score"]    = np.nan
                row[f"{topic}_attempts"] = 0
                row[f"{topic}_trend"]    = 0.0
            else:
                row[f"{topic}_score"]    = t["score"].mean()
                row[f"{topic}_attempts"] = len(t)
                row[f"{topic}_trend"]    = _slope(
                    t["quiz_number"].values,
                    t["score"].values
                )

        avg          = row["avg_score"]
        row["label"] = 0 if avg < 0.45 else (1 if avg < 0.72 else 2)
        rows.append(row)

    features = pd.DataFrame(rows)

    score_cols = [c for c in features.columns if c.endswith("_score")]
    for col in score_cols:
        features[col] = features[col].fillna(features[col].mean())

    DATA_PROC.mkdir(parents=True, exist_ok=True)
    features.to_csv(DATA_PROC / "student_features.csv", index=False)
    print(f"Features ready — {features.shape[0]} students x {features.shape[1]} columns")
    return features


def get_weak_topics(student_row: pd.Series, threshold: float = 0.5) -> list:
    return [
        topic for topic in TOPICS
        if f"{topic}_score" in student_row.index
        and student_row[f"{topic}_score"] < threshold
    ]


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    return round(float(np.polyfit(x, y, 1)[0]), 5)


if __name__ == "__main__":
    df = pd.read_csv("data/raw/quiz_history.csv")
    features = build_student_features(df)
    print(features[["student_id", "avg_score", "score_trend", "label"]].head(10).to_string(index=False))