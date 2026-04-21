"""
为静态前端(无后端)导出全部所需的"系数 + 预处理"到一个 JSON 文件。

包含:
  - SNF 4 分类:LogReg-L1 系数 / 截距 / 类别顺序 / 预处理参数
  - 生存:OS / RFS / DMFS 三个 Cox 模型 × 4 变体(base / treat / snf / snf+treat)
         系数, baseline_survival(t -> S0(t)), 预处理(数值均值/标准差, 类别 OHE 列)
  - 原文 AUC 基准
  - SNF 亚型解释

前端只需:
  1) 读 JSON
  2) 把病人 13/16 个字段做同一个预处理
  3) 矩阵相乘 + softmax / exp 即可得到完全等价的预测。
"""

from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

from data_loader import (
    ALL_FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES, TREATMENT_FEATURES,
    SNF_LABELS, SNF_DESCRIPTION, FEATURE_DESCRIPTION,
    load_table_s1, build_feature_frame, split_labeled_unlabeled, get_modeling_matrix,
)
from training import build_pipeline, cross_validate_with_ci, PAPER_BENCHMARKS
from survival import SURV_ENDPOINTS, VARIANTS, cox_four_variants

OUT_DIR = ROOT.parent / "static_app"
OUT_DIR.mkdir(exist_ok=True)


def _onehot_meta(df_col: pd.Series):
    """返回 OHE 训练时的所有类别值列表(剔除缺失;前端按这个顺序构造 0/1 向量)。"""
    return sorted(df_col.dropna().astype(str).unique().tolist())


def _serialize_classifier():
    """训出 LogReg-L1 多分类(分类: 13 个核心临床特征, 不含治疗)。"""
    raw = load_table_s1(); feats = build_feature_frame(raw)
    labeled, _ = split_labeled_unlabeled(feats)
    fset = NUMERIC_FEATURES + CATEGORICAL_FEATURES   # 13 个
    X, y = get_modeling_matrix(labeled, fset)
    numeric = [c for c in fset if c in NUMERIC_FEATURES]
    categorical = [c for c in fset if c not in NUMERIC_FEATURES]

    pipe = build_pipeline(numeric, categorical, model_name="LogReg-L1", random_state=42)
    pipe.fit(X, y)

    cv = cross_validate_with_ci(X, y, numeric, categorical,
                                model_name="LogReg-L1", n_splits=5, n_boot=400)

    pre = pipe.named_steps["preproc"]
    clf = pipe.named_steps["clf"]

    # 数值预处理参数
    num_pipe = pre.named_transformers_["num"]
    num_imputer = num_pipe.named_steps["imputer"]
    num_scaler = num_pipe.named_steps["scaler"]
    numeric_meta = {
        c: {
            "median": float(num_imputer.statistics_[i]),
            "mean":   float(num_scaler.mean_[i]),
            "std":    float(num_scaler.scale_[i]),
        } for i, c in enumerate(numeric)
    }

    # 类别预处理参数:每个原始字段下的所有 OHE 列(顺序!)
    cat_pipe = pre.named_transformers_["cat"]
    ohe = cat_pipe.named_steps["onehot"]
    categorical_meta = {}
    for i, c in enumerate(categorical):
        cats = [str(x) for x in ohe.categories_[i]]
        categorical_meta[c] = cats  # 注意:Missing 也在里面

    # 输出特征名(保持训练时顺序)
    feature_names_out = pre.get_feature_names_out().tolist()

    # 系数:shape (n_classes, n_features); LR 多分类
    coef = clf.coef_.tolist()
    intercept = clf.intercept_.tolist()
    classes = list(clf.classes_)

    return {
        "model": "LogReg-L1",
        "features": fset,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "numeric_meta": numeric_meta,
        "categorical_meta": categorical_meta,
        "feature_names_out": feature_names_out,
        "classes_": classes,
        "coef_": coef,
        "intercept_": intercept,
        "performance": {
            "macro_auc": cv.macro_auc,
            "macro_auc_ci": list(cv.macro_auc_ci),
            "weighted_auc": cv.weighted_auc,
            "weighted_auc_ci": list(cv.weighted_auc_ci),
            "per_class_auc": cv.per_class_auc,
            "per_class_auc_ci": {k: list(v) for k, v in cv.per_class_auc_ci.items()},
            "n_samples": cv.n_samples,
            "class_counts": cv.class_counts,
        },
    }


def _serialize_cox_variant(bundle, result):
    """把一个 CoxPHFitter bundle 拍平成纯数据。"""
    cph = bundle["cph"]
    pre = bundle["preprocessor"]
    used = bundle["used_columns"]
    features_orig = bundle["features_original"]

    # 数值预处理
    num_pipe = pre.named_transformers_["num"]
    num_imp = num_pipe.named_steps["imputer"]
    num_sc  = num_pipe.named_steps["scaler"]
    numeric_meta = {
        c: {
            "median": float(num_imp.statistics_[i]),
            "mean":   float(num_sc.mean_[i]),
            "std":    float(num_sc.scale_[i]),
        } for i, c in enumerate(bundle["numeric"])
    }
    cat_pipe = pre.named_transformers_["cat"]
    ohe = cat_pipe.named_steps["onehot"]
    categorical_meta = {c: [str(x) for x in ohe.categories_[i]]
                        for i, c in enumerate(bundle["categorical"])}

    # CoxPH 系数; cph.params_ 是 Series, index = 列名
    params = cph.params_.reindex(used).fillna(0.0).astype(float).tolist()

    # baseline survival: index = times, values = S0(t)
    bf = cph.baseline_survival_
    times = bf.index.astype(float).tolist()
    surv  = bf.iloc[:, 0].astype(float).tolist()

    return {
        "with_treatment": bundle["with_treatment"],
        "with_snf": bundle["with_snf"],
        "features_original": features_orig,
        "numeric": bundle["numeric"],
        "categorical": bundle["categorical"],
        "numeric_meta": numeric_meta,
        "categorical_meta": categorical_meta,
        "used_columns": used,
        "coef_": params,
        "baseline_times": times,
        "baseline_survival": surv,
        "performance": {
            "n_total": result.n_total, "n_events": result.n_events,
            "cv_c_index": result.cv_c_index,
            "cv_c_index_ci": list(result.cv_c_index_ci),
            "train_c_index": result.train_c_index,
            "test_c_index": result.test_c_index,
        },
        "top_coefficients": result.feature_coef[:10],
    }


def _serialize_survival(restrict_to_snf_labeled: bool):
    raw = load_table_s1(); df = build_feature_frame(raw)
    out = {}
    for ep in SURV_ENDPOINTS:
        variants = cox_four_variants(
            df, endpoint=ep, n_splits=5, penalizer=0.05,
            restrict_to_snf_labeled=restrict_to_snf_labeled,
        )
        ep_out = {}
        for vk, v in variants.items():
            if "error" in v:
                ep_out[vk] = {"error": v["error"], "meta": v["meta"]}
                continue
            ep_out[vk] = {**_serialize_cox_variant(v["bundle"], v["result"]),
                          "meta": v["meta"]}
        out[ep] = ep_out
    return out


def main():
    print("Training classifier (LogReg-L1)...")
    classifier = _serialize_classifier()
    print(f"  features = {len(classifier['features'])}, "
          f"weighted AUC = {classifier['performance']['weighted_auc']:.3f}")

    print("Training survival (full cohort) ...")
    surv_full = _serialize_survival(False)
    print("Training survival (matched cohort, n=350) ...")
    surv_matched = _serialize_survival(True)

    payload = {
        "version": 1,
        "data_source": "Nature Genetics 2023 (s41588-023-01507-7), Table S1",
        "labels": SNF_LABELS,
        "label_description": SNF_DESCRIPTION,
        "feature_description": FEATURE_DESCRIPTION,
        "paper_benchmarks": PAPER_BENCHMARKS,
        "classifier": classifier,
        "survival": {
            "endpoints": {ep: SURV_ENDPOINTS[ep] for ep in SURV_ENDPOINTS},
            "variants_meta": {v["key"]: v for v in VARIANTS},
            "cohorts": {"full": surv_full, "matched": surv_matched},
        },
    }

    out_path = OUT_DIR / "models.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"), default=float)
    size_kb = out_path.stat().st_size / 1024
    print(f"\n  Wrote {out_path}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
