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
    TREATMENT_FEATURES,
)

# =============================================================================
# 模型动物园(Model Zoo)
# =============================================================================
# 说明: 每个 builder 接受 random_state 返回一个 sklearn estimator(或兼容 API 的 xgb/lgbm)。
# 统一放在 build_pipeline(est=...) 里, 前端直接选。
# =============================================================================
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)


from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import LabelEncoder as _LE


class _OvRMultiBinary(BaseEstimator, ClassifierMixin):
    """
    One-vs-Rest 包装:为每一类训练一个独立二分类器(是 / 不是 SNF_i)。
    与 sklearn OneVsRestClassifier 相比,这里:
      - 每个子模型可以是任意 base 算法(通过 base_builder);
      - predict_proba 把 4 个独立 "P(is_i)" 做 softmax 归一化后返回;
      - classes_ / predict 对外完全兼容多分类接口;
      - 能单独暴露 per_class_raw_proba 方便调试。
    """
    def __init__(self, base_builder=None, random_state=42):
        self.base_builder = base_builder
        self.random_state = random_state

    def fit(self, X, y, **kw):
        y = np.asarray(y)
        self.classes_ = np.array(sorted(np.unique(y)))
        self.estimators_ = []
        for c in self.classes_:
            est = self.base_builder(self.random_state)
            y_bin = (y == c).astype(int)
            est.fit(X, y_bin, **kw)
            self.estimators_.append(est)
        return self

    def _raw_pos_proba(self, X):
        out = []
        for est in self.estimators_:
            if hasattr(est, "predict_proba"):
                p = est.predict_proba(X)
                if p.shape[1] == 2:
                    out.append(p[:, 1])
                else:
                    out.append(p[:, 0])
            elif hasattr(est, "decision_function"):
                d = est.decision_function(X)
                out.append(1.0 / (1.0 + np.exp(-d)))
            else:
                pred = est.predict(X).astype(float)
                out.append(pred)
        return np.stack(out, axis=1)

    def predict_proba(self, X):
        raw = self._raw_pos_proba(X)
        s = raw.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0
        return raw / s

    def predict(self, X):
        idx = np.argmax(self._raw_pos_proba(X), axis=1)
        return self.classes_[idx]

    def set_params(self, **kw):
        for k, v in kw.items(): setattr(self, k, v)
        return self


def _ovr_builder(base_fn):
    def _build(rs):
        return _OvRMultiBinary(base_builder=base_fn, random_state=rs)
    return _build


class _XGBWithLabelEncoder(BaseEstimator, ClassifierMixin):
    """XGBoost 要求整数标签, 这里透明地做 LabelEncoder 以便接受 SNF1-4 字符串。"""
    def __init__(self, n_estimators=500, learning_rate=0.05, max_depth=4,
                 subsample=0.9, colsample_bytree=0.9,
                 random_state=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state

    def fit(self, X, y, **kw):
        from xgboost import XGBClassifier
        self._le = _LE()
        y_enc = self._le.fit_transform(y)
        self.classes_ = self._le.classes_
        self._clf = XGBClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            objective="multi:softprob",
            tree_method="hist",
            random_state=self.random_state,
            n_jobs=-1,
            eval_metric="mlogloss",
        )
        self._clf.fit(X, y_enc, **kw)
        return self

    def predict_proba(self, X):
        return self._clf.predict_proba(X)

    def predict(self, X):
        idx = self._clf.predict(X)
        return self._le.inverse_transform(idx)

    @property
    def feature_importances_(self):
        return self._clf.feature_importances_


def _xgb(random_state):
    try:
        import xgboost  # noqa: F401
    except Exception:
        return None
    return _XGBWithLabelEncoder(
        n_estimators=500, learning_rate=0.05, max_depth=4,
        subsample=0.9, colsample_bytree=0.9,
        random_state=random_state,
    )


def _lgbm(random_state):
    try:
        from lightgbm import LGBMClassifier
    except Exception:
        return None
    return LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=31,
        subsample=0.9, colsample_bytree=0.9,
        class_weight="balanced", random_state=random_state,
        n_jobs=-1, verbose=-1,
    )


MODEL_ZOO: dict = {
    "RandomForest": lambda rs: RandomForestClassifier(
        n_estimators=500, max_depth=None, min_samples_split=3, min_samples_leaf=1,
        class_weight="balanced", random_state=rs, n_jobs=-1),
    "ExtraTrees": lambda rs: ExtraTreesClassifier(
        n_estimators=500, class_weight="balanced", random_state=rs, n_jobs=-1),
    "GradientBoosting": lambda rs: GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=3, random_state=rs),
    "HistGradientBoosting": lambda rs: HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=None, random_state=rs),
    "XGBoost": _xgb,
    "LightGBM": _lgbm,
    "LogisticRegression": lambda rs: LogisticRegression(
        max_iter=2000, C=1.0, class_weight="balanced",
        solver="lbfgs", random_state=rs),
    "LogReg-L1": lambda rs: LogisticRegression(
        max_iter=2000, C=0.5, penalty="l1", solver="saga",
        class_weight="balanced", random_state=rs),
    "LinearSVM": lambda rs: SVC(
        kernel="linear", probability=True, class_weight="balanced", random_state=rs),
    "RBF-SVM": lambda rs: SVC(
        kernel="rbf", probability=True, class_weight="balanced", random_state=rs),
    "KNN": lambda rs: KNeighborsClassifier(n_neighbors=11, weights="distance"),
    "DecisionTree": lambda rs: DecisionTreeClassifier(
        max_depth=6, class_weight="balanced", random_state=rs),
    "GaussianNB": lambda rs: GaussianNB(),
    "LDA": lambda rs: LinearDiscriminantAnalysis(),
    "QDA": lambda rs: QuadraticDiscriminantAnalysis(reg_param=0.01),
    "MLP": lambda rs: MLPClassifier(
        hidden_layer_sizes=(64, 32), max_iter=800, random_state=rs,
        early_stopping=False, alpha=1e-3),
}


# ---------------------------------------------------------------------------
# OvR(4 个独立二分类)组合
# ---------------------------------------------------------------------------
# 针对"是否 SNF_i"各训练一个二分类器, 在 predict 时 softmax 归一再 argmax。
# 与原生多分类的区别:每棵树/每条回归线只关心自己那一类, 对类别不均衡更灵活。
def _binary_rf(rs):
    return RandomForestClassifier(
        n_estimators=500, class_weight="balanced",
        min_samples_leaf=1, random_state=rs, n_jobs=-1)

def _binary_lr(rs):
    return LogisticRegression(
        max_iter=2000, C=1.0, class_weight="balanced",
        solver="lbfgs", random_state=rs)

def _binary_lr_l1(rs):
    return LogisticRegression(
        max_iter=2000, C=0.5, penalty="l1", solver="saga",
        class_weight="balanced", random_state=rs)

def _binary_svm(rs):
    return SVC(kernel="linear", probability=True,
               class_weight="balanced", random_state=rs)

def _binary_xgb(rs):
    try:
        from xgboost import XGBClassifier
    except Exception:
        return None
    return XGBClassifier(
        n_estimators=400, learning_rate=0.05, max_depth=4,
        subsample=0.9, colsample_bytree=0.9,
        objective="binary:logistic", tree_method="hist",
        random_state=rs, n_jobs=-1, eval_metric="logloss")

def _binary_lgbm(rs):
    try:
        from lightgbm import LGBMClassifier
    except Exception:
        return None
    return LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=31,
        class_weight="balanced", random_state=rs,
        n_jobs=-1, verbose=-1)


MODEL_ZOO.update({
    "OvR-LogReg":      _ovr_builder(_binary_lr),
    "OvR-LogReg-L1":   _ovr_builder(_binary_lr_l1),
    "OvR-RandomForest":_ovr_builder(_binary_rf),
    "OvR-LinearSVM":   _ovr_builder(_binary_svm),
    "OvR-XGBoost":     _ovr_builder(_binary_xgb),
    "OvR-LightGBM":    _ovr_builder(_binary_lgbm),
})


def list_available_models() -> list[str]:
    out = []
    for name, builder in MODEL_ZOO.items():
        est = builder(42)
        if est is not None:
            out.append(name)
    return out


def _strip_cat_for_numeric_only(categorical, X, numeric):
    """某些模型(GaussianNB, QDA)不擅长 sparse OHE, 但我们都过同一预处理。"""
    return categorical

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
    n_estimators: int = 500,
    random_state: int = 42,
    model_name: str = "RandomForest",
) -> Pipeline:
    """
    model_name: MODEL_ZOO 中的任一键, 默认 RandomForest。
    n_estimators 只对 RandomForest/ExtraTrees/XGB/LGBM 生效, 其余模型会忽略。
    """
    preproc = build_preprocessor(numeric, categorical)
    builder = MODEL_ZOO.get(model_name)
    if builder is None:
        raise ValueError(f"未知模型 {model_name}; 可选: {list(MODEL_ZOO)}")
    clf = builder(random_state)
    if clf is None:
        raise RuntimeError(f"{model_name} 未安装或不可用")
    # 针对树/GB 类模型覆盖树数(其它模型忽略)
    if model_name in {"RandomForest", "ExtraTrees"}:
        try: clf.set_params(n_estimators=n_estimators)
        except Exception: pass
    elif model_name in {"XGBoost", "LightGBM"}:
        try: clf.set_params(n_estimators=n_estimators)
        except Exception: pass
    elif model_name == "GradientBoosting":
        try: clf.set_params(n_estimators=min(500, max(100, n_estimators // 2)))
        except Exception: pass
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
    weighted_auc: float = float("nan")
    weighted_auc_ci: Tuple[float, float] = (float("nan"), float("nan"))
    fold_macro_auc: List[float] = field(default_factory=list)
    fold_weighted_auc: List[float] = field(default_factory=list)
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
) -> Tuple[Dict[str, Tuple[float, float]], Tuple[float, float], Tuple[float, float]]:
    """返回 (per_class_ci, macro_ci, weighted_ci)。"""
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    macro_vals = []
    weighted_vals = []
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
        try:
            weighted = roc_auc_score(yb, sb, average="weighted", multi_class="ovr")
            weighted_vals.append(weighted)
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
    weighted_ci = _q(weighted_vals)
    return per_class_ci, macro_ci, weighted_ci


def cross_validate_with_ci(
    X: pd.DataFrame,
    y: pd.Series,
    numeric: List[str],
    categorical: List[str],
    n_splits: int = 5,
    random_state: int = 42,
    n_boot: int = 500,
    n_estimators: int = 600,
    model_name: str = "RandomForest",
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
    fold_weighted = []

    for fold, (tr, va) in enumerate(skf.split(X, y_arr), 1):
        pipe = build_pipeline(numeric, categorical,
                              n_estimators=n_estimators,
                              random_state=random_state + fold,
                              model_name=model_name)
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
        try:
            weighted = roc_auc_score(y_bin, proba_sorted, average="weighted", multi_class="ovr")
            fold_weighted.append(float(weighted))
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
    weighted_auc = float(roc_auc_score(y_bin_all, oof_proba, average="weighted", multi_class="ovr"))

    per_class_ci, macro_ci, weighted_ci = bootstrap_auc_ci(
        y_arr, oof_proba, classes, n_boot=n_boot, random_state=random_state
    )

    cm = confusion_matrix(y_arr, oof_pred, labels=classes).tolist()
    report = classification_report(y_arr, oof_pred, labels=classes, digits=3, zero_division=0)

    return CVResult(
        per_class_auc=per_class,
        per_class_auc_ci=per_class_ci,
        macro_auc=macro_auc,
        macro_auc_ci=macro_ci,
        weighted_auc=weighted_auc,
        weighted_auc_ci=weighted_ci,
        fold_macro_auc=fold_macro,
        fold_weighted_auc=fold_weighted,
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
    返回 {class: {"prob":, "lo":, "hi":, "std":, "tree_agreement":}}

    - 如果是 RF/ExtraTrees 这类 bagging 模型: 用每棵树的 predict_proba 做 tree-level CI,
      CI = mean ± 1.96 · std / sqrt(n_trees)
    - 其他模型: 没有天然的树级别方差, 只返回 prob, CI 退化为 [prob, prob], std=0。
    """
    preproc = pipe.named_steps["preproc"]
    clf = pipe.named_steps["clf"]
    X_t = preproc.transform(X_new)

    estimators = getattr(clf, "estimators_", None)
    single_proba = None
    if estimators is None or not hasattr(estimators[0], "predict_proba"):
        single_proba = clf.predict_proba(X_t)[0]
        classes_fit = list(clf.classes_)
        out = {}
        for c in labels:
            if c in classes_fit:
                p = float(single_proba[classes_fit.index(c)])
            else:
                p = 0.0
            out[c] = {"prob": p, "lo": p, "hi": p, "std": 0.0, "tree_agreement": float(p >= 0.5)}
        total = sum(out[c]["prob"] for c in labels)
        if total > 0:
            for c in labels: out[c]["prob"] /= total
        return out

    tree_probas = np.stack([t.predict_proba(X_t)[0] for t in estimators], axis=0)
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
def compare_models(
    X: pd.DataFrame,
    y: pd.Series,
    numeric: List[str],
    categorical: List[str],
    model_names: Optional[List[str]] = None,
    n_splits: int = 5,
    random_state: int = 42,
    n_boot: int = 200,
    n_estimators: int = 400,
) -> List[dict]:
    """在同一份子人群/特征上并排跑多个模型, 返回按 Macro AUC 排序的结果。
    每个元素包含: name, macro_auc, macro_auc_ci, per_class_auc, per_class_auc_ci,
                  fit_ok, error(若失败), n_samples。
    """
    import time
    if model_names is None:
        model_names = list_available_models()

    results = []
    for name in model_names:
        t0 = time.time()
        rec = {"name": name}
        try:
            cv = cross_validate_with_ci(
                X, y, numeric, categorical,
                n_splits=n_splits, random_state=random_state,
                n_boot=n_boot, n_estimators=n_estimators,
                model_name=name,
            )
            rec.update({
                "macro_auc": cv.macro_auc,
                "macro_auc_ci": list(cv.macro_auc_ci),
                "weighted_auc": cv.weighted_auc,
                "weighted_auc_ci": list(cv.weighted_auc_ci),
                "per_class_auc": cv.per_class_auc,
                "per_class_auc_ci": {k: list(v) for k, v in cv.per_class_auc_ci.items()},
                "n_samples": cv.n_samples,
                "class_counts": cv.class_counts,
                "labels": cv.labels,
                "fold_macro_auc": cv.fold_macro_auc,
                "fold_weighted_auc": cv.fold_weighted_auc,
                "fit_ok": True,
                "error": None,
            })
        except Exception as e:
            rec.update({"fit_ok": False, "error": str(e),
                        "macro_auc": float("nan"),
                        "macro_auc_ci": [float("nan"), float("nan")],
                        "weighted_auc": float("nan"),
                        "weighted_auc_ci": [float("nan"), float("nan")],
                        "per_class_auc": {},
                        "per_class_auc_ci": {}})
        rec["seconds"] = round(time.time() - t0, 2)
        results.append(rec)

    # 按 weighted_auc 降序排(类别不均衡下更有代表性)
    def _sort_key(r):
        if not r["fit_ok"]:
            return (1, 0.0, r["name"])
        v = r.get("weighted_auc")
        if v is None or (isinstance(v, float) and np.isnan(v)):
            v = r.get("macro_auc", 0.0) or 0.0
        return (0, -float(v), r["name"])
    results.sort(key=_sort_key)
    return results


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
