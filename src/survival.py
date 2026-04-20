"""
生存预测(OS / RFS / DMFS)
=========================

基于 lifelines 的 Cox 比例风险回归(CoxPH):
- 共享前面的预处理(数值中位数+标准化, 类别 OneHot);
- 用 5 折交叉验证的 Concordance Index(C-index)评估泛化;
- 单独保留一个 train/test split 输出测试集 C-index;
- 对个人病人给出 24/60/120 个月的生存概率 + 完整生存曲线;
- 默认"包含术后辅助治疗"三字段(因为对生存的影响很直接),
  可以通过 with_treatment=False 关闭。

CoxPH 能天然接受"右删失"数据(OS_status=1 是发生死亡,=0 是截尾)。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from data_loader import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TREATMENT_FEATURES,
)
from training import build_preprocessor

SURV_ENDPOINTS = {
    "OS":   {"status": "OS_status",   "time": "OS_months",   "label": "总生存(OS)"},
    "RFS":  {"status": "RFS_status",  "time": "RFS_months",  "label": "无复发生存(RFS)"},
    "DMFS": {"status": "DMFS_status", "time": "DMFS_months", "label": "无远处转移生存(DMFS)"},
}


@dataclass
class SurvivalResult:
    endpoint: str
    n_total: int = 0
    n_events: int = 0
    n_train: int = 0
    n_test: int = 0
    cv_c_index: float = float("nan")
    cv_c_index_ci: Tuple[float, float] = (float("nan"), float("nan"))
    cv_folds: List[float] = field(default_factory=list)
    test_c_index: float = float("nan")
    train_c_index: float = float("nan")
    feature_coef: List[dict] = field(default_factory=list)
    baseline_times: List[float] = field(default_factory=list)
    baseline_survival: List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["cv_c_index_ci"] = list(self.cv_c_index_ci)
        return d


def build_survival_matrix(
    df: pd.DataFrame,
    features: List[str],
    endpoint: str,
    with_treatment: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, List[str], List[str]]:
    """返回 (X, T, E, numeric, categorical)。
    - X: 仅含特征列(未编码)
    - T: 事件时间(月)
    - E: 事件是否发生(1/0)
    """
    cfg = SURV_ENDPOINTS[endpoint]
    if features is None:
        features = list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)
        if with_treatment:
            features = features + list(TREATMENT_FEATURES)
    features = [f for f in features if f in ALL_FEATURES]

    keep = df.dropna(subset=[cfg["status"], cfg["time"]]).copy()
    keep = keep[keep[cfg["time"]] > 0]

    X = keep[features].copy()
    T = pd.to_numeric(keep[cfg["time"]], errors="coerce").astype(float)
    E = pd.to_numeric(keep[cfg["status"]], errors="coerce").fillna(0).astype(int)

    numeric = [c for c in features if c in NUMERIC_FEATURES]
    categorical = [c for c in features if c not in NUMERIC_FEATURES]
    return X, T, E, numeric, categorical


def _preprocess(X, numeric, categorical):
    pre = build_preprocessor(numeric, categorical)
    pre.fit(X)
    M = pre.transform(X)
    try:
        names = pre.get_feature_names_out().tolist()
    except Exception:
        names = [f"f{i}" for i in range(M.shape[1])]
    M = pd.DataFrame(M, columns=names, index=X.index if isinstance(X, pd.DataFrame) else None)
    return M, pre, names


def _drop_constant_cols(M: pd.DataFrame) -> pd.DataFrame:
    """去掉 CoxPH 无法处理的常数列(训练折里全是同一个值)。"""
    nunique = M.nunique(axis=0)
    return M.loc[:, nunique > 1]


def cox_train_test(
    df: pd.DataFrame,
    endpoint: str = "OS",
    features: Optional[List[str]] = None,
    with_treatment: bool = True,
    n_splits: int = 5,
    test_size: float = 0.25,
    random_state: int = 42,
    penalizer: float = 0.05,
) -> Tuple["object", SurvivalResult]:
    """
    同时做两件事:
    1) 5 折 CV, 报告平均 C-index 和 bootstrap-like (fold std) CI
    2) 单次 train/test(默认 75/25 分层按 E), 报告测试集 C-index
    最终模型在全量数据上再拟合一次, 用于个人预测。
    """
    from lifelines import CoxPHFitter

    X, T, E, numeric, categorical = build_survival_matrix(
        df, features, endpoint, with_treatment=with_treatment
    )
    if len(X) < 30 or E.sum() < 5:
        raise ValueError(f"{endpoint} 样本量或事件数过少 (n={len(X)}, events={int(E.sum())})")

    # ---- CV ----
    fold_c = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for tr, va in skf.split(X, E):
        M_tr, pre, _ = _preprocess(X.iloc[tr], numeric, categorical)
        M_tr = _drop_constant_cols(M_tr)
        df_tr = M_tr.copy()
        df_tr["__T"] = T.iloc[tr].values
        df_tr["__E"] = E.iloc[tr].values
        try:
            cph = CoxPHFitter(penalizer=penalizer)
            cph.fit(df_tr, duration_col="__T", event_col="__E")
        except Exception as e:
            continue
        M_va = pd.DataFrame(pre.transform(X.iloc[va]), columns=_preprocess(X.iloc[tr], numeric, categorical)[2])[M_tr.columns]
        try:
            risk = cph.predict_partial_hazard(M_va).values.ravel()
            from lifelines.utils import concordance_index
            c = concordance_index(T.iloc[va].values, -risk, E.iloc[va].values)
            fold_c.append(float(c))
        except Exception:
            continue

    cv_mean = float(np.mean(fold_c)) if fold_c else float("nan")
    cv_std  = float(np.std(fold_c, ddof=1)) if len(fold_c) > 1 else 0.0
    cv_ci = (cv_mean - 1.96 * cv_std / np.sqrt(max(len(fold_c), 1)),
             cv_mean + 1.96 * cv_std / np.sqrt(max(len(fold_c), 1)))

    # ---- Train/test split ----
    Xtr, Xte, Ttr, Tte, Etr, Ete = train_test_split(
        X, T, E, test_size=test_size, random_state=random_state, stratify=E
    )
    Mtr, pre_tt, names_tt = _preprocess(Xtr, numeric, categorical)
    Mtr = _drop_constant_cols(Mtr)
    df_tr = Mtr.copy()
    df_tr["__T"] = Ttr.values; df_tr["__E"] = Etr.values
    cph_tt = None
    train_c = test_c = float("nan")
    try:
        cph_tt = CoxPHFitter(penalizer=penalizer)
        cph_tt.fit(df_tr, duration_col="__T", event_col="__E")
        Mte = pd.DataFrame(pre_tt.transform(Xte), columns=names_tt)[Mtr.columns]
        from lifelines.utils import concordance_index
        risk_te = cph_tt.predict_partial_hazard(Mte).values.ravel()
        test_c = float(concordance_index(Tte.values, -risk_te, Ete.values))
        risk_tr = cph_tt.predict_partial_hazard(Mtr).values.ravel()
        train_c = float(concordance_index(Ttr.values, -risk_tr, Etr.values))
    except Exception:
        pass

    # ---- 最终模型(全量) ----
    M_all, pre_all, names_all = _preprocess(X, numeric, categorical)
    M_all = _drop_constant_cols(M_all)
    df_all = M_all.copy()
    df_all["__T"] = T.values; df_all["__E"] = E.values

    from lifelines import CoxPHFitter as _Cox
    cph_final = _Cox(penalizer=penalizer)
    cph_final.fit(df_all, duration_col="__T", event_col="__E")
    coefs = (
        cph_final.summary[["coef", "exp(coef)", "p"]]
        .sort_values("p").head(15).reset_index()
    )
    coef_rows = [
        {"feature": r["covariate"], "coef": float(r["coef"]),
         "hazard_ratio": float(r["exp(coef)"]), "p": float(r["p"])}
        for _, r in coefs.iterrows()
    ]

    bf = cph_final.baseline_survival_
    bf_times = bf.index.astype(float).tolist()
    bf_surv = bf.iloc[:, 0].astype(float).tolist()

    result = SurvivalResult(
        endpoint=endpoint,
        n_total=int(len(X)),
        n_events=int(E.sum()),
        n_train=int(len(Xtr)),
        n_test=int(len(Xte)),
        cv_c_index=cv_mean,
        cv_c_index_ci=cv_ci,
        cv_folds=fold_c,
        test_c_index=test_c,
        train_c_index=train_c,
        feature_coef=coef_rows,
        baseline_times=bf_times,
        baseline_survival=bf_surv,
    )

    bundle = {
        "cph": cph_final,
        "preprocessor": pre_all,
        "feature_names_preproc": names_all,
        "used_columns": M_all.columns.tolist(),  # 最终模型实际用到的列
        "features_original": list(X.columns),
        "numeric": numeric,
        "categorical": categorical,
        "endpoint": endpoint,
        "with_treatment": with_treatment,
    }
    return bundle, result


def predict_survival_curve(
    bundle: dict,
    patient: dict,
    times: Optional[List[float]] = None,
) -> Dict[str, list]:
    """
    对单个病人返回完整生存曲线以及 24/60/120 个月的生存概率 + 中位生存期(若可估计)。
    """
    cph = bundle["cph"]
    pre = bundle["preprocessor"]
    names = bundle["feature_names_preproc"]
    used = bundle["used_columns"]
    features = bundle["features_original"]

    row = {f: patient.get(f, None) for f in features}
    for c in features:
        if c in NUMERIC_FEATURES:
            row[c] = pd.to_numeric(pd.Series([row[c]]), errors="coerce").iloc[0]
    X_new = pd.DataFrame([row])
    M = pd.DataFrame(pre.transform(X_new), columns=names)[used]

    if times is None:
        maxt = float(cph.baseline_survival_.index.max())
        times = np.linspace(0.5, min(maxt, 180.0), 60).tolist()
    times = [float(t) for t in times]

    sf = cph.predict_survival_function(M, times=times).iloc[:, 0]
    med_raw = cph.predict_median(M)
    if hasattr(med_raw, "iloc"):
        med_val = med_raw.iloc[0]
    else:
        med_val = med_raw
    try:
        median = float(med_val)
        if not np.isfinite(median):
            median = None
    except Exception:
        median = None

    milestones = {}
    for t in [24, 60, 120]:
        sf_t = cph.predict_survival_function(M, times=[t]).iloc[:, 0].iloc[0]
        milestones[f"p_survive_{t}mo"] = float(sf_t)

    ph = cph.predict_partial_hazard(M)
    try:
        hazard = float(np.asarray(ph).ravel()[0])
    except Exception:
        hazard = float("nan")

    return {
        "times": times,
        "survival": [float(v) for v in sf.values],
        "median_survival_months": median,
        "milestones": milestones,
        "partial_hazard": hazard,  # > 1 表示比 baseline 风险高, < 1 更低
    }
