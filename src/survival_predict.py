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
from survival import (
    SURV_ENDPOINTS, VARIANTS,
    cox_four_variants, predict_per_subtype, predict_survival_curve,
)

OUT_DIR = ROOT.parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def _predict_snf(patient: dict):
    """用已保存的分型模型给病人一个 argmax SNF 和完整的概率向量。"""
    pkl = OUT_DIR / "snf_classifier.pkl"
    if not pkl.exists():
        return None, None
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
        prob_map = {c: float(proba[classes.index(c)]) if c in classes else 0.0
                    for c in ["SNF1","SNF2","SNF3","SNF4"]}
        return classes[int(np.argmax(proba))], prob_map
    except Exception:
        return None, None


def _plot_by_variant(endpoint_results, save_path, selected_variants):
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


def _plot_by_subtype(subtype_curves, save_path, variant_key="snf+treat"):
    """每个端点一张子图, 叠加 SNF1-4 + Expected 5 条曲线。"""
    import matplotlib.pyplot as plt
    endpoints = [ep for ep, vmap in subtype_curves.items()
                 if variant_key in vmap and "per_subtype" in vmap[variant_key]]
    if not endpoints: return
    fig, axes = plt.subplots(1, len(endpoints), figsize=(5.0 * len(endpoints), 4.2), sharey=True)
    if len(endpoints) == 1: axes = [axes]
    colors = {"SNF1":"#2563eb", "SNF2":"#059669", "SNF3":"#d97706", "SNF4":"#dc2626"}
    for ax, ep in zip(axes, endpoints):
        by = subtype_curves[ep][variant_key]
        for s in by["subtype_labels"]:
            p = by["per_subtype"][s]
            ax.plot(p["times"], p["survival"], color=colors.get(s, "gray"), lw=2, label=s)
        if "expected" in by:
            ax.plot(by["expected"]["times"], by["expected"]["survival"],
                    color="#1f2937", lw=2, ls="--", label="Expected")
        ax.set_title(f"{ep}  ·  Cox = {variant_key}")
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

    auto, snf_probs = _predict_snf(patient)
    if auto is not None:
        if not patient.get("SNF_subtype"):
            patient["SNF_subtype"] = auto
        if snf_probs is not None:
            parts = " | ".join(f"{s}={(p*100):.0f}%" for s, p in snf_probs.items())
            print(f"Tab ① 分型模型: argmax = {auto}; 概率 {parts}")

    raw = load_table_s1(); df = build_feature_frame(raw)

    print("=" * 80)
    print(f"病人: {pid}")
    print("=" * 80)

    endpoint_results = {}
    rows = []
    selected = [args.only_variant] if args.only_variant else [v["key"] for v in VARIANTS]

    subtype_curves = {}  # ep -> vk -> {SNF1: pred, SNF2: pred, ..., expected: pred}
    matched_rows = []
    for ep in SURV_ENDPOINTS.keys():
        variants = cox_four_variants(df, endpoint=ep, n_splits=args.n_splits,
                                     penalizer=args.penalizer)
        # 同时跑 matched 版本(只用 SNF 标签 cohort, 4 变体 N 一致)
        variants_matched = cox_four_variants(
            df, endpoint=ep, n_splits=args.n_splits,
            penalizer=args.penalizer, restrict_to_snf_labeled=True)
        for vk in selected:
            vm = variants_matched.get(vk)
            if vm is None or "error" in vm: continue
            r2 = vm["result"]
            matched_rows.append({
                "endpoint": ep, "variant": vk, "label": vm["meta"]["label"],
                "n_total": r2.n_total, "n_events": r2.n_events,
                "cv_c_index": r2.cv_c_index,
                "train_c_index": r2.train_c_index,
                "test_c_index": r2.test_c_index,
            })
        endpoint_results[ep] = {}
        subtype_curves[ep] = {}
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
            # 对含 SNF 的变体, 额外跑 4 种 SNF 假设
            if v["bundle"].get("with_snf"):
                try:
                    by = predict_per_subtype(v["bundle"], patient, subtype_probs=snf_probs)
                    subtype_curves[ep][vk] = by
                except Exception as e:
                    subtype_curves[ep][vk] = {"error": str(e)}

    df_out = pd.DataFrame(rows).round(3)

    # ---------- Matched cohort (n=350 一致) 公平对比 ----------
    if matched_rows:
        df_matched = pd.DataFrame(matched_rows).round(3)
        print()
        print("=" * 80)
        print("Matched cohort:只用有 SNF 标签的 ~350 例;4 变体 N 一致, 公平比较加 SNF/治疗的边际贡献")
        print("=" * 80)
        for ep, chunk in df_matched.groupby("endpoint", sort=False):
            print(f"\n-- {ep} --")
            print(chunk[["variant","label","n_total","n_events",
                         "cv_c_index","train_c_index","test_c_index"]].to_string(index=False))

    # ---------- Full cohort (各取最大样本) ----------
    print()
    print("=" * 80)
    print("Full cohort:base/treat 用全部 ~578 例; snf 系列剔除无 SNF 后 ~350 例")
    print("=" * 80)
    for ep, chunk in df_out.groupby("endpoint", sort=False):
        print()
        print(f"-------- {ep}  {SURV_ENDPOINTS[ep]['label']} --------")
        disp = chunk[["variant","label","n_total","n_events",
                      "cv_c_index","train_c_index","test_c_index",
                      "partial_hazard","p_2y","p_5y","p_10y"]]
        print(disp.to_string(index=False))

        # SNF 亚型对比
        vk_with_snf = [k for k in selected if k in subtype_curves.get(ep, {}) and "per_subtype" in subtype_curves[ep][k]]
        if vk_with_snf:
            vk = "snf+treat" if "snf+treat" in vk_with_snf else vk_with_snf[0]
            by = subtype_curves[ep][vk]
            print(f"\n  把病人假设成不同 SNF 亚型 (Cox 模型: {vk}):")
            print(f"  {'Subtype':<10}{'P(2y)':>10}{'P(5y)':>10}{'P(10y)':>10}{'HR':>8}")
            for s in by["subtype_labels"]:
                p = by["per_subtype"][s]
                m = p["milestones"]
                print(f"  {s:<10}{m['p_survive_24mo']*100:>9.1f}%{m['p_survive_60mo']*100:>9.1f}%{m['p_survive_120mo']*100:>9.1f}%{p['partial_hazard']:>8.2f}")
            if "expected" in by:
                m = by["expected"]["milestones"]
                print(f"  {'Expected':<10}{m['p_survive_24mo']*100:>9.1f}%{m['p_survive_60mo']*100:>9.1f}%{m['p_survive_120mo']*100:>9.1f}%{'--':>8}")

    df_out.to_csv(OUT_DIR / f"survival_report_{pid}.csv", index=False)
    if matched_rows:
        pd.DataFrame(matched_rows).round(3).to_csv(
            OUT_DIR / f"survival_report_{pid}_matched.csv", index=False)
    with open(OUT_DIR / f"survival_prediction_{pid}.json", "w", encoding="utf-8") as f:
        json.dump({
            "patient_id": pid,
            "auto_snf_subtype": patient.get("SNF_subtype"),
            "snf_probabilities": snf_probs,
            "by_subtype": {
                ep: {
                    vk: by for vk, by in vmap.items()
                    if "per_subtype" in by or "error" in by
                } for ep, vmap in subtype_curves.items() if vmap
            },
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

    _plot_by_variant(endpoint_results, OUT_DIR / f"survival_curve_{pid}.png",
                     selected_variants=selected)
    _plot_by_subtype(subtype_curves, OUT_DIR / f"survival_curve_{pid}_bySNF.png",
                     variant_key="snf+treat" if "snf+treat" in selected else "snf")

    print()
    print("已保存:")
    print(" -", OUT_DIR / f"survival_report_{pid}.csv")
    print(" -", OUT_DIR / f"survival_prediction_{pid}.json")
    print(" -", OUT_DIR / f"survival_curve_{pid}.png      (按变体)")
    print(" -", OUT_DIR / f"survival_curve_{pid}_bySNF.png (按 SNF 亚型假设)")


if __name__ == "__main__":
    main()
