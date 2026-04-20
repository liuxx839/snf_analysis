"""
CLI: 对每个端点(OS/RFS/DMFS)拟合 4 个 CoxPH 变体 —— 基线 / 加治疗 / 加 SNF / 加 SNF+治疗 ——
并报告 train / test / 5 折 CV 的 C-index, 以及给一个病人的个人生存曲线。

用法:
    python src/survival_predict.py --patient my_patient.yaml
    python src/survival_predict.py --patient my_patient.yaml --only-variant snf+treat
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
from survival import SURV_ENDPOINTS, VARIANTS, cox_four_variants, predict_survival_curve

OUT_DIR = ROOT.parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def _predict_snf(patient: dict):
    """尝试用已保存的 SNF 分型模型给病人打一个预测标签, 用于带 SNF 的变体。"""
    pkl = OUT_DIR / "snf_classifier.pkl"
    if not pkl.exists():
        return None
    try:
        import pickle
        with open(pkl, "rb") as f:
            bundle = pickle.load(f)
        pipe = bundle["pipeline"]
        features = bundle.get("features", [])
        from predict_patient import patient_to_row
        X = patient_to_row(patient, features)
        proba = pipe.predict_proba(X)[0]
        classes = list(pipe.classes_)
        return classes[int(np.argmax(proba))]
    except Exception:
        return None


def _plot_personal(endpoint_results, save_path, selected_variants):
    import matplotlib.pyplot as plt
    endpoints = list(endpoint_results.keys())
    fig, axes = plt.subplots(1, len(endpoints), figsize=(5.0 * len(endpoints), 4.2), sharey=True)
    if len(endpoints) == 1: axes = [axes]
    colors = {"base":"#94a3b8", "treat":"#059669", "snf":"#2563eb", "snf+treat":"#dc2626"}
    for ax, ep in zip(axes, endpoints):
        for vk in selected_variants:
            rec = endpoint_results[ep].get(vk)
            if rec is None or "error" in rec: continue
            pred = rec["pred"]; res = rec["result"]
            ax.plot(pred["times"], pred["survival"], color=colors.get(vk, "gray"), lw=2,
                    label=f"{vk} (test C={res.test_c_index:.2f})")
        ax.set_title(ep)
        ax.set_xlabel("Months"); ax.set_ylim(0, 1.02); ax.grid(alpha=0.3)
        ax.legend(loc="lower left", fontsize=8)
    axes[0].set_ylabel("Survival probability")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", default="patient_template.yaml")
    ap.add_argument("--only-variant", default=None,
                    help="只展示某一个变体(base / treat / snf / snf+treat); 默认全部 4 个")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--penalizer", type=float, default=0.05)
    args = ap.parse_args()

    with open(args.patient, "r", encoding="utf-8") as f:
        patient = yaml.safe_load(f)
    pid = patient.get("patient_id", "ME")

    if not patient.get("SNF_subtype"):
        auto = _predict_snf(patient)
        if auto is not None:
            patient["SNF_subtype"] = auto
            print(f"(Tab ① 模型自动预测 SNF_subtype = {auto}, 用于含 SNF 的变体)")

    raw = load_table_s1(); df = build_feature_frame(raw)

    print("=" * 80)
    print(f"病人: {pid}")
    print("=" * 80)

    endpoint_results = {}
    rows = []
    selected = [args.only_variant] if args.only_variant else [v["key"] for v in VARIANTS]

    for ep in SURV_ENDPOINTS.keys():
        variants = cox_four_variants(df, endpoint=ep, n_splits=args.n_splits,
                                     penalizer=args.penalizer)
        endpoint_results[ep] = {}
        for vk in selected:
            v = variants.get(vk)
            if v is None or "error" in v:
                endpoint_results[ep][vk] = {"error": (v or {}).get("error", "missing")}
                continue
            pred = predict_survival_curve(v["bundle"], patient)
            endpoint_results[ep][vk] = {"result": v["result"], "pred": pred, "meta": v["meta"]}
            res = v["result"]
            rows.append({
                "endpoint": ep, "variant": vk, "label": v["meta"]["label"],
                "n_total": res.n_total, "n_events": res.n_events,
                "cv_c_index": res.cv_c_index,
                "cv_c_lo": res.cv_c_index_ci[0], "cv_c_hi": res.cv_c_index_ci[1],
                "train_c_index": res.train_c_index,
                "test_c_index": res.test_c_index,
                "partial_hazard": pred["partial_hazard"],
                "p_2y":  pred["milestones"]["p_survive_24mo"],
                "p_5y":  pred["milestones"]["p_survive_60mo"],
                "p_10y": pred["milestones"]["p_survive_120mo"],
            })

    df_out = pd.DataFrame(rows).round(3)

    # 漂亮打印:按端点分组
    for ep, chunk in df_out.groupby("endpoint", sort=False):
        print()
        print(f"-------- {ep}  {SURV_ENDPOINTS[ep]['label']} --------")
        disp = chunk[["variant","label","n_total","n_events",
                      "cv_c_index","train_c_index","test_c_index",
                      "partial_hazard","p_2y","p_5y","p_10y"]]
        print(disp.to_string(index=False))

    df_out.to_csv(OUT_DIR / f"survival_report_{pid}.csv", index=False)
    with open(OUT_DIR / f"survival_prediction_{pid}.json", "w", encoding="utf-8") as f:
        json.dump({
            "patient_id": pid,
            "auto_snf_subtype": patient.get("SNF_subtype"),
            "endpoints": {
                ep: {
                    vk: (
                        {"error": rec["error"]} if "error" in rec else {
                            "label": rec["meta"]["label"],
                            "train_c_index": rec["result"].train_c_index,
                            "test_c_index": rec["result"].test_c_index,
                            "cv_c_index": rec["result"].cv_c_index,
                            "cv_c_index_ci": list(rec["result"].cv_c_index_ci),
                            "n_total": rec["result"].n_total,
                            "n_events": rec["result"].n_events,
                            "partial_hazard": rec["pred"]["partial_hazard"],
                            "median_survival_months": rec["pred"]["median_survival_months"],
                            "milestones": rec["pred"]["milestones"],
                            "times": rec["pred"]["times"],
                            "survival": rec["pred"]["survival"],
                            "top_coefficients": rec["result"].feature_coef[:8],
                        }
                    )
                    for vk, rec in vs.items()
                }
                for ep, vs in endpoint_results.items()
            },
        }, f, indent=2, ensure_ascii=False, default=float)

    _plot_personal(endpoint_results, OUT_DIR / f"survival_curve_{pid}.png",
                   selected_variants=selected)

    print()
    print("已保存:")
    print(" -", OUT_DIR / f"survival_report_{pid}.csv")
    print(" -", OUT_DIR / f"survival_prediction_{pid}.json")
    print(" -", OUT_DIR / f"survival_curve_{pid}.png")


if __name__ == "__main__":
    main()
