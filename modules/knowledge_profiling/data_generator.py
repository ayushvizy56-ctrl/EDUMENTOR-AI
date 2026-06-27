import numpy as np
import pandas as pd
from config.settings import TOPICS, DATA_RAW, N_STUDENTS, QUIZZES_EACH, RANDOM_STATE


def generate_dataset() -> pd.DataFrame:
    np.random.seed(RANDOM_STATE)
    records = []

    for student_id in range(N_STUDENTS):
        base_ability = np.random.beta(a=2, b=2)

        topic_strengths = {
            topic: np.clip(base_ability + np.random.normal(0, 0.15), 0.05, 0.99)
            for topic in TOPICS
        }

        for quiz_num in range(1, QUIZZES_EACH + 1):
            topic      = np.random.choice(TOPICS)
            difficulty = np.random.uniform(0.2, 0.9)
            struggle   = 1 - topic_strengths[topic]
            time_spent = np.clip(np.random.normal(30 + struggle * 20, 10), 5, 120)
            hints_used = np.random.poisson(lam=struggle * 2)
            p_correct  = topic_strengths[topic] - (difficulty * 0.3) + (hints_used * 0.04)
            p_correct  = np.clip(p_correct, 0.05, 0.97)
            score      = round(np.random.binomial(n=10, p=p_correct) / 10, 1)

            records.append({
                "student_id":     student_id,
                "quiz_number":    quiz_num,
                "topic":          topic,
                "difficulty":     round(difficulty, 2),
                "time_spent_sec": round(time_spent, 1),
                "hints_used":     max(0, hints_used),
                "score":          score,
            })

    df = pd.DataFrame(records)
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_RAW / "quiz_history.csv", index=False)
    print(f"Done — {len(df):,} rows for {N_STUDENTS} students")
    return df


if __name__ == "__main__":
    df = generate_dataset()
    print(df.head(8).to_string(index=False))