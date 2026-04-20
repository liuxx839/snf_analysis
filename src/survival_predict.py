"""
CLI: 训练 OS / RFS / DMFS 三个 CoxPH 生存模型, 报告 train/test/CV 的 C-index,
并为一个病人 YAML 给出 24/60/120 个月生存概率 + 个人生存曲线图。

用法:
    python src/survival_predict.py --patient my_patient.yaml
    python src/survival_predict.py --patient my_patient.yaml --no-treatment
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data_loader import load_table_s1, build_feature_frame
from survival import SURV_ENDPOINTS, cox_train_test, predict_survival_curve

OUT_DIR = ROOT.parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def _plot_personal_curves(results, save_path):
    import matplotlib.pyplot as plt
    colors = {"OS": "#2563eb", "RFS": "#059669", "DMFS": "#d97706"}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for ep, (res, pred) in results.items():
        ax.plot(pred["times"], pred["survival"], color=colors.get(ep, "gray"), lw=2,
                label=f"{ep}  test C={res.test_c_index:.2f}")
    ax.set_xlabel("Months")
    ax.set_ylabel("Survival probability")
    ax.set_ylim(0, 1.02)
    ax.set_title("Personal survival curves (CoxPH)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", default="patient_template.yaml")
    ap.add_argument("--no-treatment", dest="with_treatment", action="store_false")
    ap.add_argument("--with-treatment", dest="with_treatment", action="store_true")
    ap.set_defaults(with_treatment=True)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--penalizer", type=float, default=0.05)
    args = ap.parse_args()

    with open(args.patient, "r", encoding="utf-8") as f:
        patient = yaml.safe_load(f)
    pid = patient.get("patient_id", "ME")

    raw = load_table_s1()
    df = build_feature_frame(raw)

    print("=" * 70)
    print(f"病人: {pid}   |   辅助治疗: {'含' if args.with_treatment else '不含'}")
    print("=" * 70)

    rows = []
    personal = {}
    for ep in SURV_ENDPOINTS.keys():
        bundle, res = cox_train_test(
            df, endpoint=ep,
            with_treatment=args.with_treatment,
            n_splits=args.n_splits,
            penalizer=args.penalizer,
        )
        pred = predict_survival_curve(bundle, patient)
        rows.append({
            "endpoint": ep,
            "label": SURV_ENDPOINTS[ep]["label"],
            "n_total": res.n_total, "n_events": res.n_events,
            "cv_c_index": res.cv_c_index,
            "cv_c_index_lo": res.cv_c_index_ci[0],
            "cv_c_index_hi": res.cv_c_index_ci[1],
            "train_c_index": res.train_c_index,
            "test_c_index": res.test_c_index,
            "partial_hazard": pred["partial_hazard"],
            "p_2y":  pred["milestones"]["p_survive_24mo"],
            "p_5y":  pred["milestones"]["p_survive_60mo"],
            "p_10y": pred["milestones"]["p_survive_120mo"],
        })
        personal[ep] = (res, pred)

    df_out = pd.DataFrame(rows)
    print()
    print(df_out.round(3).to_string(index=False))

    df_out.to_csv(OUT_DIR / f"survival_report_{pid}.csv", index=False)
    with open(OUT_DIR / f"survival_prediction_{pid}.json", "w", encoding="utf-8") as f:
        json.dump({
            "patient_id": pid,
            "with_treatment": args.with_treatment,
            "endpoints": {ep: {
                "label": SURV_ENDPOINTS[ep]["label"],
                "n_total": res.n_total, "n_events": res.n_events,
                "cv_c_index": res.cv_c_index, "cv_c_index_ci": list(res.cv_c_index_ci),
                "train_c_index": res.train_c_index,
                "test_c_index": res.test_c_index,
                "partial_hazard": pred["partial_hazard"],
                "median_survival_months": pred["median_survival_months"],
                "milestones": pred["milestones"],
                "times": pred["times"],
                "survival": pred["survival"],
                "top_coefficients": res.feature_coef[:8],
            } for ep, (res, pred) in personal.items()},
        }, f, indent=2, ensure_ascii=False, default=float)

    _plot_personal_curves(personal, OUT_DIR / f"survival_curve_{pid}.png")
    print()
    print("已保存:")
    print(" -", OUT_DIR / f"survival_report_{pid}.csv")
    print(" -", OUT_DIR / f"survival_prediction_{pid}.json")
    print(" -", OUT_DIR / f"survival_curve_{pid}.png")


if __name__ == "__main__":
    main()
