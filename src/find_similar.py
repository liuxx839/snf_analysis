"""
在 Table S1 中找到与你最相似的病人, 并对比生存(OS / RFS / DMFS)。
相似度: 在标准化(数值) + One-Hot(类别) 后的欧氏距离, 距离越小越相似。
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import yaml

from data_loader import (
    ALL_FEATURES,
    NUMERIC_FEATURES,
    SNF_LABELS,
    build_feature_frame,
    get_modeling_matrix,
    load_table_s1,
    split_labeled_unlabeled,
)
from predict_patient import patient_to_row, load_patient

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
MODEL_PATH = OUT_DIR / "snf_classifier.pkl"


def transform_with_pipeline(pipe, X: pd.DataFrame) -> np.ndarray:
    preproc = pipe.named_steps["preproc"]
    return preproc.transform(X)


def find_similar(patient_yaml: Path, k: int = 10, restrict_predicted_subtype: bool = False):
    raw = load_table_s1()
    feats = build_feature_frame(raw)
    labeled, _ = split_labeled_unlabeled(feats)
    X_all, y_all = get_modeling_matrix(labeled, ALL_FEATURES)

    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    pipe = bundle["pipeline"]

    X_all_t = transform_with_pipeline(pipe, X_all)

    patient = load_patient(patient_yaml)
    patient_id = patient.get("patient_id", "ME")
    X_new = patient_to_row(patient)
    X_new_t = transform_with_pipeline(pipe, X_new)

    dists = np.linalg.norm(X_all_t - X_new_t, axis=1)

    proba = pipe.predict_proba(X_new)[0]
    labels = bundle["labels"]
    col_idx = {c: list(pipe.classes_).index(c) for c in labels}
    proba_sorted = np.array([proba[col_idx[c]] for c in labels])
    pred = labels[int(np.argmax(proba_sorted))]

    df_sim = labeled.copy()
    df_sim["distance"] = dists
    df_sim["similarity_rank"] = df_sim["distance"].rank(method="first").astype(int)

    if restrict_predicted_subtype:
        df_sim = df_sim[df_sim["SNF_subtype"] == pred]

    df_sim = df_sim.sort_values("distance").head(k)

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
    out = df_sim[show_cols].copy()
    out.index.name = "PatientCode"

    print("=" * 90)
    print(f"病人 {patient_id} 预测 SNF 亚型: {pred}  概率分布: " + ", ".join(f"{c}={p:.2f}" for c,p in zip(labels, proba_sorted)))
    print(f"在原队列(N={len(labeled)})中,临床特征空间下最相似的 Top {k}{'(只看同亚型)' if restrict_predicted_subtype else ''}:")
    print("=" * 90)
    with pd.option_context("display.max_rows", None, "display.max_columns", None,
                           "display.width", 200):
        print(out.to_string(float_format=lambda v: f"{v:.2f}"))

    print("\n---- 相似病人事件汇总 ----")
    for evt, status_col, time_col in [
        ("OS(总生存)", "OS_status", "OS_months"),
        ("RFS(无复发生存)", "RFS_status", "RFS_months"),
        ("DMFS(无远处转移)", "DMFS_status", "DMFS_months"),
    ]:
        if status_col in out.columns and time_col in out.columns:
            n = out[status_col].notna().sum()
            events = int(out[status_col].fillna(0).sum())
            med = out[time_col].median()
            print(f"  {evt}: 有随访 {n} 人, 发生事件 {events} 人, 中位随访 {med:.1f} 月")

    print("\n---- 相似病人 SNF 分布 ----")
    print(out["SNF_subtype"].value_counts().to_string())

    OUT_DIR.mkdir(exist_ok=True)
    csv_path = OUT_DIR / f"similar_patients_{patient_id}.csv"
    out.to_csv(csv_path)
    print(f"\n相似病人列表已保存: {csv_path}")

    return out, pred, dict(zip(labels, proba_sorted.tolist()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", type=str, default="patient_template.yaml")
    ap.add_argument("--k", type=int, default=15, help="返回最相似的前 K 名")
    ap.add_argument("--same-subtype-only", action="store_true",
                    help="只在预测的亚型内找相似病人")
    args = ap.parse_args()

    find_similar(Path(args.patient), k=args.k, restrict_predicted_subtype=args.same_subtype_only)


if __name__ == "__main__":
    main()
