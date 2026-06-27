import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
from config.settings import MODEL_DIR, TEST_SIZE, RANDOM_STATE

LABELS = {0: "Weak", 1: "Average", 2: "Strong"}


def train():
    df = pd.read_csv("data/processed/student_features.csv")
    X  = df.drop(columns=["student_id", "label"])
    y  = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    print("Training models...\n")

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=12,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_train_sc, y_train)

    xgb = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.08,
        subsample=0.85, colsample_bytree=0.85,
        eval_metric="mlogloss", random_state=RANDOM_STATE,
        n_jobs=-1, verbosity=0
    )
    xgb.fit(X_train_sc, y_train)

    rf_acc  = rf.score(X_test_sc, y_test)
    xgb_acc = xgb.score(X_test_sc, y_test)
    print(f"Random Forest : {rf_acc:.4f}")
    print(f"XGBoost       : {xgb_acc:.4f}")

    best, name = (xgb, "XGBoost") if xgb_acc >= rf_acc else (rf, "Random Forest")
    print(f"\nWinner: {name}\n")

    y_pred = best.predict(X_test_sc)
    print(classification_report(y_test, y_pred, target_names=list(LABELS.values())))

    cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(best, X_train_sc, y_train, cv=cv)
    print(f"5-Fold CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABELS.values(),
                yticklabels=LABELS.values(), ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {name}")
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "confusion_matrix.png", dpi=120)
    plt.close()

    joblib.dump(best,               MODEL_DIR / "knowledge_model.pkl")
    joblib.dump(scaler,             MODEL_DIR / "scaler.pkl")
    joblib.dump(X.columns.tolist(), MODEL_DIR / "feature_names.pkl")
    print(f"\nAll files saved to {MODEL_DIR}/")


if __name__ == "__main__":
    train()