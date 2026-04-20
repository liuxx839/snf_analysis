"""
画 SNF1-4 在原队列里的 KM 生存曲线(OS/RFS/DMFS), 并把"相似病人 vs 全队列"
的 KM 单独画一张, 方便对照预后。
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

from data_loader import (
    ALL_FEATURES,
    SNF_LABELS,
    build_feature_frame,
    get_modeling_matrix,
    load_table_s1,
    split_labeled_unlabeled,
)
from find_similar import find_similar  # 复用

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def km_by_subtype(df: pd.DataFrame, status_col: str, time_col: str, title: str, save_path: Path):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    kmf = KaplanMeierFitter()
    sub = df[[status_col, time_col, "SNF_subtype"]].dropna()
    for s in SNF_LABELS:
        m = sub["SNF_subtype"] == s
        if m.sum() < 5:
            continue
        kmf.fit(sub.loc[m, time_col], sub.loc[m, status_col], label=f"{s} (n={int(m.sum())})")
        kmf.plot_survival_function(ax=ax, ci_show=False)
    try:
        lr = multivariate_logrank_test(sub[time_col], sub["SNF_subtype"], sub[status_col])
        p = lr.p_value
        ax.set_title(f"{title}  (log-rank p = {p:.3g})")
    except Exception:
        ax.set_title(title)
    ax.set_xlabel("Months")
    ax.set_ylabel("Survival probability")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def km_similar_vs_cohort(similar: pd.DataFrame, cohort: pd.DataFrame,
                         status_col: str, time_col: str, title: str, save_path: Path):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    kmf = KaplanMeierFitter()

    sim = similar[[status_col, time_col]].dropna()
    coh = cohort[[status_col, time_col]].dropna()

    if len(coh) >= 5:
        kmf.fit(coh[time_col], coh[status_col], label=f"Whole cohort (n={len(coh)})")
        kmf.plot_survival_function(ax=ax, ci_show=False)
    if len(sim) >= 3:
        kmf.fit(sim[time_col], sim[status_col], label=f"Top-K similar patients (n={len(sim)})")
        kmf.plot_survival_function(ax=ax, ci_show=False)
    ax.set_title(title)
    ax.set_xlabel("Months")
    ax.set_ylabel("Survival probability")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", type=str, default="patient_template.yaml")
    ap.add_argument("--k", type=int, default=20)
    args = ap.parse_args()

    raw = load_table_s1()
    feats = build_feature_frame(raw)
    labeled, _ = split_labeled_unlabeled(feats)

    OUT_DIR.mkdir(exist_ok=True)
    for st, tm, ttl, fn in [
        ("OS_status", "OS_months", "Overall Survival by SNF subtype", "km_OS_by_SNF.png"),
        ("RFS_status", "RFS_months", "RFS by SNF subtype", "km_RFS_by_SNF.png"),
        ("DMFS_status", "DMFS_months", "DMFS by SNF subtype", "km_DMFS_by_SNF.png"),
    ]:
        km_by_subtype(labeled, st, tm, ttl, OUT_DIR / fn)

    sim, pred, probs = find_similar(Path(args.patient), k=args.k, restrict_predicted_subtype=False)
    pid = "ME"
    try:
        import yaml
        with open(args.patient) as f:
            pid = yaml.safe_load(f).get("patient_id", "ME")
    except Exception:
        pass

    for st, tm, ttl, fn in [
        ("OS_status", "OS_months", f"OS: similar Top-{args.k} vs whole cohort", f"km_OS_similar_{pid}.png"),
        ("RFS_status", "RFS_months", f"RFS: similar Top-{args.k} vs whole cohort", f"km_RFS_similar_{pid}.png"),
        ("DMFS_status", "DMFS_months", f"DMFS: similar Top-{args.k} vs whole cohort", f"km_DMFS_similar_{pid}.png"),
    ]:
        km_similar_vs_cohort(sim, labeled, st, tm, ttl, OUT_DIR / fn)

    print("已保存 KM 图到", OUT_DIR)


if __name__ == "__main__":
    main()
