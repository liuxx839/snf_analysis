"""
SNF 四分型分类模型
==================

基于 Table S1 的临床特征训练一个多分类器, 用于在没有转录组/病理切片的情况下
给出 SNF1-SNF4 的概率预测, 并报告交叉验证 AUC。

模型: Random Forest(与原文 transcriptomics 分类器同族, 对混合类型特征稳健)。
    还会同时保存一个 XGBoost 版本作为对比(如可用)。

策略:
    - 数值特征: 中位数填补 + 标准化
    - 类别特征: 常数"Missing"填补 + One-Hot
    - 管道式 Pipeline, 以便在对新病人推理时复现完全相同的预处理。
    - 交叉验证: 分层 5 折, 报告 one-vs-rest macro AUC 与各类别 AUC。
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_loader import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    SNF_LABELS,
    build_feature_frame,
    get_modeling_matrix,
    load_table_s1,
    split_labeled_unlabeled,
)

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("onehot", ohe),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ]
    )


def build_pipeline(random_state: int = 42) -> Pipeline:
    preproc = build_preprocessor()
    clf = RandomForestClassifier(
        n_estimators=800,
        max_depth=None,
        min_samples_split=3,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    return Pipeline(steps=[("preproc", preproc), ("clf", clf)])


@dataclass
class CVResult:
    per_class_auc: Dict[str, float]
    macro_auc: float
    fold_macro_auc: List[float] = field(default_factory=list)
    classification_report: str = ""
    confusion_matrix: List[List[int]] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    oof_proba: np.ndarray | None = None
    oof_true: np.ndarray | None = None

    def to_dict(self) -> dict:
        return {
            "per_class_auc": self.per_class_auc,
            "macro_auc": self.macro_auc,
            "fold_macro_auc": self.fold_macro_auc,
            "classification_report": self.classification_report,
            "confusion_matrix": self.confusion_matrix,
            "labels": self.labels,
        }


def cross_validate(
    X: pd.DataFrame, y: pd.Series, n_splits: int = 5, random_state: int = 42
) -> CVResult:
    pipe = build_pipeline(random_state)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    classes = SNF_LABELS
    oof_proba = np.zeros((len(y), len(classes)))
    oof_pred = np.empty(len(y), dtype=object)
    fold_macro = []

    y_arr = y.to_numpy()
    for fold, (tr, va) in enumerate(skf.split(X, y_arr), 1):
        pipe_f = build_pipeline(random_state + fold)
        pipe_f.fit(X.iloc[tr], y_arr[tr])
        proba = pipe_f.predict_proba(X.iloc[va])
        col_order = [list(pipe_f.classes_).index(c) for c in classes]
        proba_sorted = proba[:, col_order]
        oof_proba[va] = proba_sorted
        oof_pred[va] = pipe_f.predict(X.iloc[va])

        y_bin = np.stack([(y_arr[va] == c).astype(int) for c in classes], axis=1)
        macro = roc_auc_score(y_bin, proba_sorted, average="macro", multi_class="ovr")
        fold_macro.append(float(macro))

    per_class = {}
    for i, c in enumerate(classes):
        y_bin = (y_arr == c).astype(int)
        per_class[c] = float(roc_auc_score(y_bin, oof_proba[:, i]))

    y_bin_all = np.stack([(y_arr == c).astype(int) for c in classes], axis=1)
    macro_auc = float(roc_auc_score(y_bin_all, oof_proba, average="macro", multi_class="ovr"))

    cm = confusion_matrix(y_arr, oof_pred, labels=classes).tolist()
    report = classification_report(y_arr, oof_pred, labels=classes, digits=3)

    return CVResult(
        per_class_auc=per_class,
        macro_auc=macro_auc,
        fold_macro_auc=fold_macro,
        classification_report=report,
        confusion_matrix=cm,
        labels=list(classes),
        oof_proba=oof_proba,
        oof_true=y_arr,
    )


def fit_final(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> Pipeline:
    pipe = build_pipeline(random_state)
    pipe.fit(X, y)
    return pipe


def plot_roc(oof_true: np.ndarray, oof_proba: np.ndarray, classes: List[str], save_path: Path):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 6))
    for i, c in enumerate(classes):
        y_bin = (oof_true == c).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, oof_proba[:, i])
        auc = roc_auc_score(y_bin, oof_proba[:, i])
        plt.plot(fpr, tpr, lw=2, label=f"{c} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("5-fold CV ROC (clinical-feature SNF classifier)")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_confusion(cm: List[List[int]], classes: List[str], save_path: Path):
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm_arr = np.array(cm)
    plt.figure(figsize=(5, 4.5))
    sns.heatmap(
        cm_arr, annot=True, fmt="d", cmap="Blues",
        xticklabels=classes, yticklabels=classes, cbar=False
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("CV Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def feature_importance(pipe: Pipeline, top_k: int = 20) -> pd.DataFrame:
    preproc: ColumnTransformer = pipe.named_steps["preproc"]
    clf: RandomForestClassifier = pipe.named_steps["clf"]
    try:
        names = preproc.get_feature_names_out()
    except Exception:
        names = np.array([f"f{i}" for i in range(len(clf.feature_importances_))])
    imp = pd.DataFrame({"feature": names, "importance": clf.feature_importances_})
    return imp.sort_values("importance", ascending=False).head(top_k).reset_index(drop=True)


def main():
    raw = load_table_s1()
    feats = build_feature_frame(raw)
    labeled, _ = split_labeled_unlabeled(feats)
    X, y = get_modeling_matrix(labeled, ALL_FEATURES)

    print("样本数:", len(X), "特征数:", X.shape[1])
    print("SNF 分布:")
    print(y.value_counts().to_string())

    cv = cross_validate(X, y, n_splits=5, random_state=42)
    print("\n==== 5 折交叉验证 AUC (one-vs-rest) ====")
    for k, v in cv.per_class_auc.items():
        print(f"  {k}: AUC = {v:.3f}")
    print(f"  Macro AUC: {cv.macro_auc:.3f}  (folds: {', '.join(f'{x:.3f}' for x in cv.fold_macro_auc)})")
    print("\n==== 交叉验证分类报告 ====")
    print(cv.classification_report)

    plot_roc(cv.oof_true, cv.oof_proba, cv.labels, OUT_DIR / "roc_cv.png")
    plot_confusion(cv.confusion_matrix, cv.labels, OUT_DIR / "confusion_matrix_cv.png")

    final_model = fit_final(X, y)
    with open(OUT_DIR / "snf_classifier.pkl", "wb") as f:
        pickle.dump(
            {
                "pipeline": final_model,
                "features": ALL_FEATURES,
                "numeric_features": NUMERIC_FEATURES,
                "categorical_features": CATEGORICAL_FEATURES,
                "labels": SNF_LABELS,
                "cv_metrics": cv.to_dict(),
            },
            f,
        )
    with open(OUT_DIR / "cv_metrics.json", "w") as f:
        json.dump(cv.to_dict(), f, indent=2, ensure_ascii=False)

    imp = feature_importance(final_model, top_k=20)
    imp.to_csv(OUT_DIR / "feature_importance_top20.csv", index=False)
    print("\nTop 10 重要特征:")
    print(imp.head(10).to_string(index=False))

    print("\n已保存:")
    print(" -", OUT_DIR / "snf_classifier.pkl")
    print(" -", OUT_DIR / "cv_metrics.json")
    print(" -", OUT_DIR / "roc_cv.png")
    print(" -", OUT_DIR / "confusion_matrix_cv.png")
    print(" -", OUT_DIR / "feature_importance_top20.csv")


if __name__ == "__main__":
    main()
