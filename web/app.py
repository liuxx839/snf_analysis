"""
FastAPI 后端:提供
  GET  /              -> 主页 (index.html)
  GET  /api/meta      -> 队列元信息(可用字段的候选值/范围)
  POST /api/train     -> 根据子人群 + 特征子集训练,返回 CV AUC + CI
  POST /api/predict   -> 用最近一次训练好的模型(或默认模型)预测病人,带预测 CI
  POST /api/similar   -> 找最相似 Top-K 病人,支持是否按特征重要性加权
  GET  /api/benchmarks-> 原文 transcriptomics RF / pathology CNN 基准
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from data_loader import (  # noqa: E402
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TREATMENT_FEATURES,
    FEATURE_DESCRIPTION,
    SNF_DESCRIPTION,
    SNF_LABELS,
    build_feature_frame,
    get_modeling_matrix,
    load_table_s1,
    split_labeled_unlabeled,
)
from training import (  # noqa: E402
    MODEL_ZOO,
    PAPER_BENCHMARKS,
    apply_subpopulation,
    build_pipeline,
    compare_models,
    compute_similarity,
    cross_validate_with_ci,
    list_available_models,
    predict_with_ci,
)
from survival import (  # noqa: E402
    SURV_ENDPOINTS,
    VARIANTS as SURV_VARIANTS,
    cox_four_variants,
    cox_train_test,
    predict_per_subtype,
    predict_survival_curve,
)

app = FastAPI(title="SNF subtype predictor", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# 启动时加载队列
# ---------------------------------------------------------------------------
print("Loading cohort ...")
_raw = load_table_s1()
_feats = build_feature_frame(_raw)
_labeled, _ = split_labeled_unlabeled(_feats)
print(f"  labeled N = {len(_labeled)}")

# 最近一次训练的模型(session 级,非持久)
_state: dict = {
    "pipeline": None,
    "numeric": list(NUMERIC_FEATURES),
    "categorical": list(CATEGORICAL_FEATURES),
    "features": list(ALL_FEATURES),
    "labels": list(SNF_LABELS),
    "filters": {},
    "train_df": None,
    "cv": None,
}


_DEFAULT_FEATURES = list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)  # 默认不含治疗


def _default_train_if_needed():
    if _state["pipeline"] is None:
        features = _DEFAULT_FEATURES
        numeric = [c for c in features if c in NUMERIC_FEATURES]
        categorical = [c for c in features if c not in NUMERIC_FEATURES]
        X, y = get_modeling_matrix(_labeled, features)
        pipe = build_pipeline(numeric, categorical,
                              n_estimators=500, random_state=42,
                              model_name="RandomForest")
        pipe.fit(X, y)
        _state["pipeline"] = pipe
        _state["features"] = features
        _state["numeric"] = numeric
        _state["categorical"] = categorical
        _state["model_name"] = "RandomForest"
        _state["train_df"] = _labeled.copy()


# ---------------------------------------------------------------------------
# 请求 / 响应模型
# ---------------------------------------------------------------------------
class TrainRequest(BaseModel):
    features: list[str] | None = Field(default=None, description="使用哪些特征, None=全部")
    filters: dict | None = Field(default=None, description="子人群过滤")
    n_splits: int = 5
    n_boot: int = 400
    n_estimators: int = 500
    random_state: int = 42
    model_name: str = "RandomForest"


class CompareRequest(BaseModel):
    features: list[str] | None = None
    filters: dict | None = None
    model_names: list[str] | None = None
    n_splits: int = 5
    n_boot: int = 200
    n_estimators: int = 300
    random_state: int = 42
    auto_select: bool = True


class PredictRequest(BaseModel):
    patient: dict


class SimilarRequest(BaseModel):
    patient: dict
    k: int = 15
    same_subtype_only: bool = False
    weight_by_importance: bool = True


class SurvivalTrainRequest(BaseModel):
    endpoints: list[str] | None = None  # 默认 ["OS","RFS","DMFS"]
    features: list[str] | None = None
    n_splits: int = 5
    penalizer: float = 0.05
    random_state: int = 42
    # 跑全部 4 个变体(基线 / 治疗 / SNF / SNF+治疗),这样前端能一次看到对比
    run_variants: bool = True


class SurvivalPredictRequest(BaseModel):
    patient: dict
    endpoints: list[str] | None = None     # 若为空则使用已训练的所有端点
    variants: list[str] | None = None      # 若为空则返回全部 4 个变体的预测
    default_variant: str = "snf+treat"
    cohort: str = "full"                   # "full" or "matched"


def _patient_to_row(patient: dict, features: list[str]) -> pd.DataFrame:
    row = {f: patient.get(f, None) for f in features}
    df = pd.DataFrame([row])
    for c in NUMERIC_FEATURES:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in CATEGORICAL_FEATURES + TREATMENT_FEATURES:
        if c in df.columns:
            df[c] = df[c].astype("object").where(df[c].notna(), None)
            df[c] = df[c].apply(lambda v: None if v is None or str(v).strip() == "" else str(v))
    return df


# ---------------------------------------------------------------------------
# /api/meta : 用于前端自动生成表单/筛选器
# ---------------------------------------------------------------------------
@app.get("/api/meta")
def meta():
    labeled = _labeled
    meta_cols: dict = {}
    for c in NUMERIC_FEATURES:
        s = pd.to_numeric(labeled[c], errors="coerce")
        meta_cols[c] = {
            "type": "numeric",
            "min": float(np.nanmin(s)) if s.notna().any() else None,
            "max": float(np.nanmax(s)) if s.notna().any() else None,
            "median": float(np.nanmedian(s)) if s.notna().any() else None,
            "missing_pct": float(s.isna().mean() * 100),
        }
    for c in CATEGORICAL_FEATURES + TREATMENT_FEATURES:
        vc = labeled[c].astype(str).replace({"nan": None}).dropna().value_counts()
        meta_cols[c] = {
            "type": "categorical",
            "values": vc.index.tolist(),
            "counts": vc.values.tolist(),
            "missing_pct": float(labeled[c].isna().mean() * 100),
        }

    sub_dist = labeled["SNF_subtype"].value_counts().to_dict()
    return {
        "n_labeled": int(len(labeled)),
        "n_total": int(len(_feats)),
        "features": ALL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "treatment_features": TREATMENT_FEATURES,
        "feature_description": FEATURE_DESCRIPTION,
        "labels": SNF_LABELS,
        "label_description": SNF_DESCRIPTION,
        "columns": meta_cols,
        "subtype_distribution": {k: int(v) for k, v in sub_dist.items()},
        "available_models": list_available_models(),
    }


# ---------------------------------------------------------------------------
# /api/benchmarks
# ---------------------------------------------------------------------------
@app.get("/api/benchmarks")
def benchmarks():
    return {
        "per_class": PAPER_BENCHMARKS,
        "macro": {
            name: float(np.mean(list(vals.values())))
            for name, vals in PAPER_BENCHMARKS.items()
        },
        "note": (
            "数据来自 Nature 2023 s41588-023-01507-7 (Gong et al.): "
            "Transcriptomics RF 与 Pathology CNN 在原文交叉验证中的 one-vs-rest AUC。"
        ),
    }


# ---------------------------------------------------------------------------
# /api/train
# ---------------------------------------------------------------------------
@app.post("/api/train")
def train(req: TrainRequest):
    features = req.features or ALL_FEATURES
    features = [f for f in features if f in ALL_FEATURES]
    if not features:
        raise HTTPException(400, "features 为空")
    numeric = [c for c in features if c in NUMERIC_FEATURES]
    categorical = [c for c in features if c not in NUMERIC_FEATURES]

    if req.model_name not in MODEL_ZOO:
        raise HTTPException(400, f"未知模型 {req.model_name}")

    df_sub = apply_subpopulation(_labeled, req.filters or {})
    if len(df_sub) < 20:
        raise HTTPException(400, f"子人群样本过少 (N={len(df_sub)}),请放宽筛选条件。")

    X, y = get_modeling_matrix(df_sub, features)

    try:
        cv = cross_validate_with_ci(
            X, y, numeric, categorical,
            n_splits=req.n_splits,
            random_state=req.random_state,
            n_boot=req.n_boot,
            n_estimators=req.n_estimators,
            model_name=req.model_name,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

    pipe = build_pipeline(numeric, categorical,
                          n_estimators=req.n_estimators,
                          random_state=req.random_state,
                          model_name=req.model_name)
    pipe.fit(X, y)

    _state["pipeline"] = pipe
    _state["numeric"] = numeric
    _state["categorical"] = categorical
    _state["features"] = features
    _state["labels"] = cv.labels
    _state["filters"] = req.filters or {}
    _state["train_df"] = df_sub.copy()
    _state["cv"] = cv
    _state["model_name"] = req.model_name

    try:
        names = pipe.named_steps["preproc"].get_feature_names_out().tolist()
    except Exception:
        names = None
    clf = pipe.named_steps["clf"]
    imp_values = None
    if hasattr(clf, "feature_importances_"):
        try: imp_values = clf.feature_importances_.tolist()
        except Exception: imp_values = None
    elif hasattr(clf, "coef_"):
        try:
            coef = clf.coef_
            imp_values = np.abs(coef).mean(axis=0).tolist() if coef.ndim == 2 else np.abs(coef).tolist()
        except Exception: imp_values = None
    if imp_values is None or names is None:
        top = []
    else:
        top = sorted(zip(names, imp_values), key=lambda x: -x[1])[:15]

    roc_points = {}
    from sklearn.metrics import roc_curve
    for i, c in enumerate(cv.labels):
        y_bin = (cv.oof_true == c).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, cv.oof_proba[:, i])
        roc_points[c] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(),
                         "auc": cv.per_class_auc[c],
                         "auc_ci": list(cv.per_class_auc_ci[c])}

    benchmarks_overlay = {
        name: {c: PAPER_BENCHMARKS[name].get(c) for c in cv.labels}
        for name in PAPER_BENCHMARKS
    }

    return {
        "n_samples": cv.n_samples,
        "class_counts": cv.class_counts,
        "labels": cv.labels,
        "features_used": features,
        "filters_applied": _state["filters"],
        "model_name": req.model_name,
        "macro_auc": cv.macro_auc,
        "macro_auc_ci": list(cv.macro_auc_ci),
        "weighted_auc": cv.weighted_auc,
        "weighted_auc_ci": list(cv.weighted_auc_ci),
        "per_class_auc": cv.per_class_auc,
        "per_class_auc_ci": {k: list(v) for k, v in cv.per_class_auc_ci.items()},
        "fold_macro_auc": cv.fold_macro_auc,
        "fold_weighted_auc": cv.fold_weighted_auc,
        "classification_report": cv.classification_report,
        "confusion_matrix": cv.confusion_matrix,
        "feature_importance_top15": [{"name": n, "importance": float(v)} for n, v in top],
        "roc_points": roc_points,
        "paper_benchmarks": benchmarks_overlay,
    }


# ---------------------------------------------------------------------------
# /api/compare : 同时比较多个模型
# ---------------------------------------------------------------------------
@app.post("/api/compare")
def compare(req: CompareRequest):
    features = req.features or ALL_FEATURES
    features = [f for f in features if f in ALL_FEATURES]
    if not features:
        raise HTTPException(400, "features 为空")
    numeric = [c for c in features if c in NUMERIC_FEATURES]
    categorical = [c for c in features if c not in NUMERIC_FEATURES]

    df_sub = apply_subpopulation(_labeled, req.filters or {})
    if len(df_sub) < 20:
        raise HTTPException(400, f"子人群样本过少 (N={len(df_sub)}),请放宽筛选条件。")

    X, y = get_modeling_matrix(df_sub, features)

    results = compare_models(
        X, y, numeric, categorical,
        model_names=req.model_names,
        n_splits=req.n_splits,
        random_state=req.random_state,
        n_boot=req.n_boot,
        n_estimators=req.n_estimators,
    )

    best_name = None
    auto_selected = False
    if req.auto_select:
        ok = [r for r in results if r["fit_ok"]]
        if ok:
            best_name = ok[0]["name"]
            best_cv = cross_validate_with_ci(
                X, y, numeric, categorical,
                n_splits=req.n_splits, random_state=req.random_state,
                n_boot=req.n_boot, n_estimators=req.n_estimators,
                model_name=best_name,
            )
            pipe = build_pipeline(numeric, categorical,
                                  n_estimators=req.n_estimators,
                                  random_state=req.random_state,
                                  model_name=best_name)
            pipe.fit(X, y)
            _state["pipeline"] = pipe
            _state["numeric"] = numeric
            _state["categorical"] = categorical
            _state["features"] = features
            _state["labels"] = best_cv.labels
            _state["filters"] = req.filters or {}
            _state["train_df"] = df_sub.copy()
            _state["cv"] = best_cv
            _state["model_name"] = best_name
            auto_selected = True

    return {
        "n_samples": int(len(y)),
        "features_used": features,
        "filters_applied": req.filters or {},
        "results": results,
        "best_model": best_name,
        "auto_selected_as_current": auto_selected,
        "paper_benchmarks": {
            name: {c: PAPER_BENCHMARKS[name].get(c) for c in SNF_LABELS}
            for name in PAPER_BENCHMARKS
        },
    }


# ---------------------------------------------------------------------------
# /api/predict
# ---------------------------------------------------------------------------
@app.post("/api/predict")
def predict(req: PredictRequest):
    _default_train_if_needed()
    pipe = _state["pipeline"]
    features = _state["features"]
    labels = _state["labels"]

    X_new = _patient_to_row(req.patient, features)
    result = predict_with_ci(pipe, X_new, labels)

    pred = max(result.items(), key=lambda kv: kv[1]["prob"])[0]
    second = sorted(result.items(), key=lambda kv: -kv[1]["prob"])[1][0]
    top_prob = result[pred]["prob"]
    margin = top_prob - result[second]["prob"]
    if top_prob >= 0.55 and margin >= 0.2:
        confidence = "high"
    elif top_prob >= 0.4 and margin >= 0.1:
        confidence = "medium"
    else:
        confidence = "low"

    interpretation_parts = [
        f"模型最可能的亚型为 {pred}(概率 {top_prob:.2f}),",
        f"95% 置信区间约 [{result[pred]['lo']:.2f}, {result[pred]['hi']:.2f}]。",
    ]
    if confidence == "high":
        interpretation_parts.append("预测较为确定。")
    elif confidence == "medium":
        interpretation_parts.append(f"但与 {second} 差距不算大,建议结合临床谨慎参考。")
    else:
        interpretation_parts.append(
            f"与 {second} 差距很小,属于边界病例——仅靠临床特征难以拍板,需要转录组/病理图像补充。"
        )
    interpretation_parts.append(SNF_DESCRIPTION.get(pred, ""))

    cv = _state.get("cv")
    model_perf = None
    if cv is not None:
        model_perf = {
            "macro_auc": cv.macro_auc,
            "macro_auc_ci": list(cv.macro_auc_ci),
            "weighted_auc": cv.weighted_auc,
            "weighted_auc_ci": list(cv.weighted_auc_ci),
            "per_class_auc": cv.per_class_auc,
            "per_class_auc_ci": {k: list(v) for k, v in cv.per_class_auc_ci.items()},
            "n_samples": cv.n_samples,
        }

    return {
        "predicted_subtype": pred,
        "confidence": confidence,
        "probabilities": result,
        "interpretation": " ".join(interpretation_parts),
        "subtype_description": {c: SNF_DESCRIPTION.get(c, "") for c in labels},
        "features_used": features,
        "filters_applied": _state.get("filters", {}),
        "model_name": _state.get("model_name", "RandomForest"),
        "model_performance": model_perf,
    }


# ---------------------------------------------------------------------------
# /api/similar
# ---------------------------------------------------------------------------
@app.post("/api/similar")
def similar(req: SimilarRequest):
    _default_train_if_needed()
    pipe = _state["pipeline"]
    features = _state["features"]
    labels = _state["labels"]

    train_df = _state["train_df"]
    X_all, y_all = get_modeling_matrix(train_df, features)
    X_new = _patient_to_row(req.patient, features)

    dists = compute_similarity(pipe, X_all, X_new,
                               weight_by_importance=req.weight_by_importance)

    proba = pipe.predict_proba(X_new)[0]
    class_map = {c: list(pipe.classes_).index(c) for c in labels if c in pipe.classes_}
    pred = max(class_map.keys(), key=lambda c: proba[class_map[c]])

    df_sim = train_df.copy()
    df_sim["distance"] = dists
    if req.same_subtype_only:
        df_sim = df_sim[df_sim["SNF_subtype"] == pred]
    df_sim = df_sim.sort_values("distance").head(req.k)

    show_cols = [
        "SNF_subtype", "PAM50", "Age", "Menopause", "Grade",
        "Tumor_size_cm", "Positive_axillary_lymph_nodes", "pT", "pN",
        "ER_percent", "PR_percent", "Ki67", "HER2_IHC_Status",
        "OS_status", "OS_months",
        "RFS_status", "RFS_months",
        "DMFS_status", "DMFS_months",
        "distance",
    ]
    show_cols = [c for c in show_cols if c in df_sim.columns]
    sub = df_sim[show_cols].copy()
    sub.insert(0, "PatientCode", sub.index.astype(str))
    sub = sub.where(pd.notna(sub), None)

    def _safe(v):
        if isinstance(v, (np.floating,)):
            v = float(v)
        if isinstance(v, float):
            return None if (np.isnan(v) or np.isinf(v)) else v
        if isinstance(v, (np.integer,)):
            return int(v)
        return v

    rows = [{k: _safe(v) for k, v in rec.items()} for rec in sub.to_dict(orient="records")]

    def _surv_summary(df, status_col, time_col):
        if status_col not in df.columns or time_col not in df.columns:
            return None
        s = df[[status_col, time_col]].dropna()
        if s.empty:
            return None
        return {
            "n": int(len(s)),
            "events": int(s[status_col].sum()),
            "median_months": float(s[time_col].median()),
        }

    summary = {
        "OS":   _surv_summary(df_sim, "OS_status",   "OS_months"),
        "RFS":  _surv_summary(df_sim, "RFS_status",  "RFS_months"),
        "DMFS": _surv_summary(df_sim, "DMFS_status", "DMFS_months"),
    }
    subtype_dist = df_sim["SNF_subtype"].value_counts().to_dict()

    return {
        "predicted_subtype": pred,
        "k": len(rows),
        "rows": rows,
        "columns": ["PatientCode"] + show_cols,
        "survival_summary": summary,
        "subtype_distribution": {k: int(v) for k, v in subtype_dist.items()},
        "weight_by_importance": req.weight_by_importance,
    }


# ---------------------------------------------------------------------------
# 生存预测(CoxPH)
# ---------------------------------------------------------------------------
# _surv_state: {
#   "full":    {endpoint: {variant_key: {"bundle":..., "result":..., "meta":...}}},
#   "matched": {...同上, 但只用 SNF 标签 cohort, n 一致...}
# }
_surv_state: dict = {"full": {}, "matched": {}}


def _variant_entry_to_dict(entry):
    """把一个 variant 的训练产物序列化成前端能读的 dict。"""
    if "error" in entry:
        return {"error": entry["error"], "meta": entry["meta"]}
    res = entry["result"]
    return {
        "meta": entry["meta"],
        "n_total": res.n_total,
        "n_events": res.n_events,
        "n_train": res.n_train,
        "n_test": res.n_test,
        "cv_c_index": res.cv_c_index,
        "cv_c_index_ci": list(res.cv_c_index_ci),
        "train_c_index": res.train_c_index,
        "test_c_index": res.test_c_index,
        "top_coefficients": res.feature_coef[:8],
    }


@app.post("/api/survival/train")
def survival_train(req: SurvivalTrainRequest):
    endpoints = req.endpoints or list(SURV_ENDPOINTS.keys())
    out_full = {}
    out_matched = {}
    for ep in endpoints:
        # full cohort:base/treat 用全部样本, snf 系列自动剔除
        variants_full = cox_four_variants(
            _feats, endpoint=ep,
            features=req.features,
            n_splits=req.n_splits,
            penalizer=req.penalizer,
            random_state=req.random_state,
        )
        # matched cohort:全部 4 个变体都只用 SNF 标签子集, n 一致
        variants_matched = cox_four_variants(
            _feats, endpoint=ep,
            features=req.features,
            n_splits=req.n_splits,
            penalizer=req.penalizer,
            random_state=req.random_state,
            restrict_to_snf_labeled=True,
        )
        _surv_state["full"][ep] = variants_full
        _surv_state["matched"][ep] = variants_matched
        out_full[ep] = {
            "label": SURV_ENDPOINTS[ep]["label"],
            "variants": {k: _variant_entry_to_dict(v) for k, v in variants_full.items()},
        }
        out_matched[ep] = {
            "label": SURV_ENDPOINTS[ep]["label"],
            "variants": {k: _variant_entry_to_dict(v) for k, v in variants_matched.items()},
        }
    return {
        "variants_order": [v["key"] for v in SURV_VARIANTS],
        "variants_meta": {v["key"]: v for v in SURV_VARIANTS},
        "cohorts": {
            "full": {
                "description": "Full cohort: base/treat 用全部样本(~578); snf 系列自动剔除无 SNF (~350). 用最大可用样本量评估。",
                "endpoints": out_full,
            },
            "matched": {
                "description": "Matched cohort: 4 个变体都只用 SNF 标签 cohort (~350), N 一致. 公平对比'加 SNF / 加治疗' 的边际贡献。",
                "endpoints": out_matched,
            },
        },
        # 兼容老前端字段(等同于 full):
        "endpoints": out_full,
    }


@app.post("/api/survival/predict")
def survival_predict(req: SurvivalPredictRequest):
    cohort_state = _surv_state.get(req.cohort)
    if not cohort_state:
        raise HTTPException(400, "尚未训练生存模型, 请先调用 /api/survival/train。")
    endpoints = req.endpoints or list(cohort_state.keys())
    wanted_variants = req.variants or [v["key"] for v in SURV_VARIANTS]

    patient = dict(req.patient)

    # 用 Tab ① 分型模型计算 SNF 的概率分布, 供含 SNF 的变体:
    #   - 4 条 per-subtype 曲线(SNF1/2/3/4 各一条假设曲线)
    #   - 一条按概率加权的 expected 曲线
    #   - argmax 作为点估计用于 base/treat 变体
    snf_probs: Dict[str, float] | None = None
    auto_snf: str | None = None
    try:
        _default_train_if_needed()
        pipe = _state["pipeline"]; feats = _state["features"]
        X_new = _patient_to_row(patient, feats)
        proba = pipe.predict_proba(X_new)[0]
        classes = list(pipe.classes_)
        snf_probs = {c: float(proba[classes.index(c)]) if c in classes else 0.0
                     for c in ["SNF1","SNF2","SNF3","SNF4"]}
        if patient.get("SNF_subtype"):
            auto_snf = patient["SNF_subtype"]
        else:
            auto_snf = max(snf_probs, key=snf_probs.get)
            patient["SNF_subtype"] = auto_snf
    except Exception:
        pass

    out = {}
    for ep in endpoints:
        vmap = cohort_state.get(ep)
        if vmap is None:
            out[ep] = {"error": "该端点尚未训练"}
            continue
        variant_preds = {}
        for vk in wanted_variants:
            entry = vmap.get(vk)
            if entry is None or "error" in entry:
                if entry is not None:
                    variant_preds[vk] = {"error": entry["error"], "meta": entry["meta"]}
                continue
            try:
                # 所有变体都生成 4 种 SNF 假设曲线;
                # base/treat 不用 SNF, 4 条会重合 -- 正好作为"加 SNF 之前"的对照。
                by_subtype = None
                try:
                    by_subtype = predict_per_subtype(
                        entry["bundle"], patient,
                        subtype_probs=snf_probs,
                    )
                except Exception as ee:
                    by_subtype = {"error": str(ee)}
                pred = predict_survival_curve(entry["bundle"], patient)
                res = entry["result"]
                variant_preds[vk] = {
                    "meta": entry["meta"],
                    "prediction": pred,
                    "by_subtype": by_subtype,
                    "uses_snf": bool(entry["bundle"].get("with_snf")),
                    "performance": {
                        "cv_c_index": res.cv_c_index,
                        "cv_c_index_ci": list(res.cv_c_index_ci),
                        "train_c_index": res.train_c_index,
                        "test_c_index": res.test_c_index,
                        "n_total": res.n_total,
                        "n_events": res.n_events,
                    },
                    "top_coefficients": res.feature_coef[:8],
                }
            except Exception as e:
                variant_preds[vk] = {"error": str(e), "meta": entry["meta"]}
        out[ep] = {
            "label": SURV_ENDPOINTS[ep]["label"],
            "variants": variant_preds,
        }
    return {
        "default_variant": req.default_variant,
        "cohort": req.cohort,
        "variants_order": [v["key"] for v in SURV_VARIANTS],
        "variants_meta": {v["key"]: v for v in SURV_VARIANTS},
        "auto_predicted_snf": auto_snf,
        "snf_probabilities": snf_probs,
        "endpoints": out,
    }


@app.get("/api/survival/status")
def survival_status():
    return {
        "trained_endpoints": list(_surv_state.keys()),
        "available_endpoints": list(SURV_ENDPOINTS.keys()),
        "endpoint_labels": {k: v["label"] for k, v in SURV_ENDPOINTS.items()},
        "variants": [v["key"] for v in SURV_VARIANTS],
    }


# ---------------------------------------------------------------------------
# 静态文件
# ---------------------------------------------------------------------------
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web.app:app", host="0.0.0.0", port=8000, reload=False)
