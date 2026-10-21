import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline


def evaluate_cv(
    name: str, pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, cv: StratifiedKFold
) -> dict:
    """Run cross-validation and return train and validation AUC."""

    scores = cross_validate(
        pipeline, X, y, cv=cv, scoring="roc_auc", return_train_score=True
    )
    train_auc = scores["train_score"].mean()
    cv_auc = scores["test_score"].mean()
    cv_std = scores["test_score"].std()

    print(
        f"{name}\nTrain AUC: {train_auc:.4f} | CV AUC: {cv_auc:.4f} (+/- {cv_std:.4f})  |  Gini: {2 * cv_auc - 1:.4f}"
    )
    print(f"CV AUC per fold: {scores['test_score'].round(4)}\n")

    return {
        "Model": name,
        "Train AUC": round(train_auc, 4),
        "CV AUC": round(cv_auc, 4),
        "Std": round(cv_std, 4),
        "Gini": round(2 * cv_auc - 1, 3),
    }
