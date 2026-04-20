"""
可复用的训练/评估模块
======================

- 支持自定义特征子集、子人群(subpopulation)过滤、样本权重。
- 训练时给出 5 折 CV 的 AUC + bootstrap 95% 置信区间。
- 预测时用 RandomForest 每棵树的概率方差估计预测不确定度(95% CI)。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

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
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_loader import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    SNF_LABELS,
)

# =============================================================================
# 原文基准(来自 Nature 2023 s41588-023-01507-7 正文/扩展图)
# =============================================================================
PAPER_BENCHMARKS: Dict[str, Dict[str, float]] = {
    "Transcriptomics RF": {"SNF1": 0.95, "SNF2": 0.93, "SNF3": 0.85, "SNF4": 0.82},
    "Pathology CNN":      {"SNF1": 0.87, "SNF2": 0.81, "SNF3": 0.78, "SNF4": 0.78},
}


# =============================================================================
# Pipeline
# =============================================================================
def build_preprocessor(numeric: List[str], categorical: List[str]) -> ColumnTransformer:
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", ohe),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric),
            ("cat", cat_pipe, categorical),
        ]
    )


def build_pipeline(
    numeric: List[str],
    categorical: List[str],
    n_estimators: int = 800,
    random_state: int = 42,
) -> Pipeline:
    preproc = build_preprocessor(numeric, categorical)
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_split=3,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    return Pipeline([("preproc", preproc), ("clf", clf)])


# =============================================================================
# 子人群过滤
# =============================================================================
def apply_subpopulation(
    df: pd.DataFrame, filters: Optional[Dict[str, list]] = None
) -> pd.DataFrame:
    """
    filters 举例:
        {"Menopause": ["Yes"], "Grade": [2, 3], "pN": ["pN0","pN1"]}
    数值字段用字典 {"min":..., "max":...} 代替 list。
    """
    if not filters:
        return df
    mask = pd.Series(True, index=df.index)
    for col, spec in filters.items():
        if col not in df.columns:
            continue
        if isinstance(spec, dict):
            lo, hi = spec.get("min"), spec.get("max")
            if lo is not None:
                mask &= pd.to_numeric(df[col], errors="coerce") >= lo
            if hi is not None:
                mask &= pd.to_numeric(df[col], errors="coerce") <= hi
        else:
            values = [str(v) for v in spec]
            mask &= df[col].astype(str).isin(values)
    return df[mask]


# =============================================================================
# 交叉验证 + bootstrap CI
# =============================================================================
@dataclass
class CVResult:
    per_class_auc: Dict[str, float] = field(default_factory=dict)
    per_class_auc_ci: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    macro_auc: float = float("nan")
    macro_auc_ci: Tuple[float, float] = (float("nan"), float("nan"))
    fold_macro_auc: List[float] = field(default_factory=list)
    classification_report: str = ""
    confusion_matrix: List[List[int]] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    n_samples: int = 0
    class_counts: Dict[str, int] = field(default_factory=dict)
    oof_proba: Optional[np.ndarray] = None
    oof_true: Optional[np.ndarray] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("oof_proba", None)
        d.pop("oof_true", None)
        return d


def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    classes: List[str],
    n_boot: int = 500,
    alpha: float = 0.05,
    random_state: int = 42,
) -> Tuple[Dict[str, Tuple[float, float]], Tuple[float, float]]:
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    macro_vals = []
    per_vals: Dict[str, list] = {c: [] for c in classes}

    y_bin = np.stack([(y_true == c).astype(int) for c in classes], axis=1)

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y_bin[idx]
        sb = y_score[idx]
        try:
            macro = roc_auc_score(yb, sb, average="macro", multi_class="ovr")
            macro_vals.append(macro)
        except ValueError:
            pass
        for i, c in enumerate(classes):
            if yb[:, i].sum() == 0 or yb[:, i].sum() == n:
                continue
            try:
                per_vals[c].append(roc_auc_score(yb[:, i], sb[:, i]))
            except ValueError:
                pass

    def _q(vals):
        if not vals:
            return (float("nan"), float("nan"))
        lo = float(np.percentile(vals, 100 * alpha / 2))
        hi = float(np.percentile(vals, 100 * (1 - alpha / 2)))
        return (lo, hi)

    per_class_ci = {c: _q(per_vals[c]) for c in classes}
    macro_ci = _q(macro_vals)
    return per_class_ci, macro_ci


def cross_validate_with_ci(
    X: pd.DataFrame,
    y: pd.Series,
    numeric: List[str],
    categorical: List[str],
    n_splits: int = 5,
    random_state: int = 42,
    n_boot: int = 500,
    n_estimators: int = 600,
) -> CVResult:
    y_arr = y.to_numpy()
    classes = [c for c in SNF_LABELS if (y_arr == c).sum() >= n_splits]
    if len(classes) < 2:
        raise ValueError("可用类别少于 2,无法训练多分类模型。请放宽子人群过滤。")

    mask_keep = np.isin(y_arr, classes)
    X = X[mask_keep].reset_index(drop=True) if isinstance(X, pd.DataFrame) else X[mask_keep]
    y_arr = y_arr[mask_keep]

    min_class_n = min((y_arr == c).sum() for c in classes)
    effective_splits = max(2, min(n_splits, int(min_class_n)))

    skf = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=random_state)

    oof_proba = np.zeros((len(y_arr), len(classes)))
    oof_pred = np.empty(len(y_arr), dtype=object)
    fold_macro = []

    for fold, (tr, va) in enumerate(skf.split(X, y_arr), 1):
        pipe = build_pipeline(numeric, categorical,
                              n_estimators=n_estimators,
                              random_state=random_state + fold)
        pipe.fit(X.iloc[tr] if isinstance(X, pd.DataFrame) else X[tr], y_arr[tr])
        proba = pipe.predict_proba(X.iloc[va] if isinstance(X, pd.DataFrame) else X[va])
        col_order = [list(pipe.classes_).index(c) if c in pipe.classes_ else -1 for c in classes]
        proba_sorted = np.zeros((len(va), len(classes)))
        for j, col in enumerate(col_order):
            if col >= 0:
                proba_sorted[:, j] = proba[:, col]
        row_sum = proba_sorted.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        proba_sorted = proba_sorted / row_sum

        oof_proba[va] = proba_sorted
        oof_pred[va] = np.array(classes)[np.argmax(proba_sorted, axis=1)]

        y_bin = np.stack([(y_arr[va] == c).astype(int) for c in classes], axis=1)
        try:
            macro = roc_auc_score(y_bin, proba_sorted, average="macro", multi_class="ovr")
            fold_macro.append(float(macro))
        except ValueError:
            pass

    per_class = {}
    for i, c in enumerate(classes):
        yb = (y_arr == c).astype(int)
        if yb.sum() > 0 and yb.sum() < len(yb):
            per_class[c] = float(roc_auc_score(yb, oof_proba[:, i]))
        else:
            per_class[c] = float("nan")

    y_bin_all = np.stack([(y_arr == c).astype(int) for c in classes], axis=1)
    macro_auc = float(roc_auc_score(y_bin_all, oof_proba, average="macro", multi_class="ovr"))

    per_class_ci, macro_ci = bootstrap_auc_ci(
        y_arr, oof_proba, classes, n_boot=n_boot, random_state=random_state
    )

    cm = confusion_matrix(y_arr, oof_pred, labels=classes).tolist()
    report = classification_report(y_arr, oof_pred, labels=classes, digits=3, zero_division=0)

    return CVResult(
        per_class_auc=per_class,
        per_class_auc_ci=per_class_ci,
        macro_auc=macro_auc,
        macro_auc_ci=macro_ci,
        fold_macro_auc=fold_macro,
        classification_report=report,
        confusion_matrix=cm,
        labels=list(classes),
        n_samples=int(len(y_arr)),
        class_counts={c: int((y_arr == c).sum()) for c in classes},
        oof_proba=oof_proba,
        oof_true=y_arr,
    )


# =============================================================================
# 预测 + 森林方差置信区间
# =============================================================================
def predict_with_ci(
    pipe: Pipeline,
    X_new: pd.DataFrame,
    labels: List[str],
    alpha: float = 0.05,
) -> Dict[str, Dict[str, float]]:
    """
    用随机森林每棵树的概率(tree-level)做 bootstrap-like 置信区间。
    返回 {class: {"prob":, "lo":, "hi":, "std":}}
    """
    preproc = pipe.named_steps["preproc"]
    clf: RandomForestClassifier = pipe.named_steps["clf"]
    X_t = preproc.transform(X_new)

    tree_probas = np.stack([t.predict_proba(X_t)[0] for t in clf.estimators_], axis=0)
    classes_fit = list(clf.classes_)
    n_trees = tree_probas.shape[0]

    z = 1.959963984540054 if math.isclose(alpha, 0.05) else 1.959963984540054

    out = {}
    for c in labels:
        if c in classes_fit:
            col = classes_fit.index(c)
            vals = tree_probas[:, col]
        else:
            vals = np.zeros(n_trees)
        mean = float(vals.mean())
        std = float(vals.std(ddof=1)) if n_trees > 1 else 0.0
        # 用森林均值的标准误构造 CI:
        #   mean ± 1.96 * std / sqrt(n_trees)
        # 这反映"如果换一批同规模的树,估计到的概率会波动多少"。
        se = std / math.sqrt(n_trees) if n_trees > 0 else 0.0
        lo = max(0.0, mean - z * se)
        hi = min(1.0, mean + z * se)
        out[c] = {
            "prob": mean,
            "lo": lo,
            "hi": hi,
            "std": std,
            "tree_agreement": float((vals > 0.5).mean()),
        }

    total = sum(out[c]["prob"] for c in labels)
    if total > 0:
        for c in labels:
            out[c]["prob"] = out[c]["prob"] / total
    return out


# =============================================================================
# 相似病人(可加特征权重)
# =============================================================================
def compute_similarity(
    pipe: Pipeline,
    X_train: pd.DataFrame,
    X_query: pd.DataFrame,
    weight_by_importance: bool = False,
) -> np.ndarray:
    preproc = pipe.named_steps["preproc"]
    T_train = preproc.transform(X_train)
    T_query = preproc.transform(X_query)

    if weight_by_importance:
        clf = pipe.named_steps["clf"]
        w = np.asarray(clf.feature_importances_, dtype=float)
        if w.sum() > 0:
            w = w / w.sum() * len(w)
            T_train = T_train * np.sqrt(w)
            T_query = T_query * np.sqrt(w)

    dists = np.linalg.norm(T_train - T_query, axis=1)
    return dists
