"""
训练 SNF 四分型分类模型(命令行版)
====================================

与 Web 前端使用同一套 training.py:支持任意模型 + bootstrap 95% CI。

用法:
    # 默认: 跑 16 个模型大比拼, 自动选 Macro AUC 最高的那个保存
    python src/model.py

    # 指定一个算法, 不做比拼
    python src/model.py --model LogReg-L1

    # 只比几个
    python src/model.py --compare RandomForest XGBoost LogReg-L1 LinearSVM LDA

    # 不含辅助治疗变量 (--no-treatment), 只用核心临床 (默认)
    python src/model.py --no-treatment

    # 也把术后辅助治疗三字段算进去
    python src/model.py --with-treatment
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_loader import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    SNF_LABELS,
    TREATMENT_FEATURES,
    build_feature_frame,
    get_modeling_matrix,
    load_table_s1,
    split_labeled_unlabeled,
)
from training import (
    PAPER_BENCHMARKS,
    build_pipeline,
    compare_models,
    cross_validate_with_ci,
    list_available_models,
)

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def _plot_roc(oof_true: np.ndarray, oof_proba: np.ndarray, labels, save_path: Path, title: str):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score, roc_curve

    plt.figure(figsize=(6, 6))
    for i, c in enumerate(labels):
        y_bin = (oof_true == c).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, oof_proba[:, i])
        auc = roc_auc_score(y_bin, oof_proba[:, i])
        plt.plot(fpr, tpr, lw=2, label=f"{c} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def _plot_confusion(cm, classes, save_path: Path):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(5, 4.5))
    sns.heatmap(
        np.array(cm), annot=True, fmt="d", cmap="Blues",
        xticklabels=classes, yticklabels=classes, cbar=False,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("CV Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def _plot_compare_bars(results, save_path: Path):
    import matplotlib.pyplot as plt

    ok = [r for r in results if r["fit_ok"]]
    if not ok:
        return
    names = [r["name"] for r in ok]
    aucs = [r["macro_auc"] for r in ok]
    lo = [r["macro_auc_ci"][0] for r in ok]
    hi = [r["macro_auc_ci"][1] for r in ok]
    err_lo = np.array(aucs) - np.array(lo)
    err_hi = np.array(hi) - np.array(aucs)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(names))
    ax.bar(x, aucs, yerr=[err_lo, err_hi], capsize=4, color="#93c5fd", edgecolor="#1d4ed8")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=40, ha="right")
    ax.set_ylabel("Macro AUC (5-fold CV, bootstrap 95% CI)")
    ax.set_ylim(0.5, 1.0)
    ax.grid(axis="y", alpha=0.3)

    for name, color in [("Transcriptomics RF", "#16a34a"), ("Pathology CNN", "#f59e0b")]:
        vals = list(PAPER_BENCHMARKS[name].values())
        macro = sum(vals) / len(vals)
        ax.axhline(macro, color=color, linestyle="--", lw=1.5,
                   label=f"{name} (paper macro={macro:.2f})")
    ax.legend(loc="lower right")
    ax.set_title("Model comparison on clinical features")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def _save_bundle(pipe, cv, features, model_name):
    bundle_path = OUT_DIR / "snf_classifier.pkl"
    with open(bundle_path, "wb") as f:
        pickle.dump({
            "pipeline": pipe,
            "features": features,
            "numeric_features": [c for c in features if c in NUMERIC_FEATURES],
            "categorical_features": [c for c in features if c not in NUMERIC_FEATURES],
            "labels": cv.labels,
            "model_name": model_name,
            "cv_metrics": cv.to_dict(),
        }, f)
    with open(OUT_DIR / "cv_metrics.json", "w") as f:
        json.dump({
            "model_name": model_name,
            "features": features,
            **cv.to_dict(),
        }, f, indent=2, ensure_ascii=False)
    return bundle_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="直接指定一个算法(见 --list-models);不指定则跑大比拼并自动选最佳。")
    ap.add_argument("--compare", nargs="+", default=None,
                    help="只比较这些模型;默认比较全部。")
    ap.add_argument("--list-models", action="store_true", help="列出可用算法后退出。")
    ap.add_argument("--with-treatment", action="store_true",
                    help="加入术后辅助治疗三字段作为特征(治疗端变量,默认不加)。")
    ap.add_argument("--no-treatment", dest="with_treatment", action="store_false")
    ap.set_defaults(with_treatment=False)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--n-estimators", type=int, default=500)
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()

    if args.list_models:
        print("Available models:")
        for m in list_available_models():
            print("  -", m)
        return

    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    if args.with_treatment:
        features = features + TREATMENT_FEATURES
    numeric = [c for c in features if c in NUMERIC_FEATURES]
    categorical = [c for c in features if c not in NUMERIC_FEATURES]

    raw = load_table_s1()
    feats = build_feature_frame(raw)
    labeled, _ = split_labeled_unlabeled(feats)
    X, y = get_modeling_matrix(labeled, features)

    print(f"样本 N = {len(X)}, 特征 {len(features)} (治疗字段: {'含' if args.with_treatment else '不含'})")
    print("SNF 分布:")
    print(y.value_counts().to_string())
    print()

    # ---------- 模型大比拼(可选) ----------
    compared = None
    if args.model is None:
        names = args.compare or list_available_models()
        print(f"运行模型大比拼 ({len(names)} 个算法, n_splits={args.n_splits}, n_boot={args.n_boot})...")
        compared = compare_models(
            X, y, numeric, categorical,
            model_names=names,
            n_splits=args.n_splits,
            random_state=args.random_state,
            n_boot=args.n_boot,
            n_estimators=args.n_estimators,
        )
        print("\n==== Leaderboard ====")
        print(f"{'Rank':<5}{'Model':<22}{'Macro AUC':>11}{'95% CI':>18}{'time(s)':>9}")
        for i, r in enumerate(compared, 1):
            if r["fit_ok"]:
                ci = f"[{r['macro_auc_ci'][0]:.2f}, {r['macro_auc_ci'][1]:.2f}]"
                print(f"{i:<5}{r['name']:<22}{r['macro_auc']:>11.3f}{ci:>18}{r['seconds']:>9}")
            else:
                print(f"{i:<5}{r['name']:<22}   FAILED  {r['error'][:70]}")

        # 保存排行
        pd.DataFrame([
            {"rank": i + 1, "name": r["name"], "fit_ok": r["fit_ok"],
             "macro_auc": r.get("macro_auc"),
             "macro_auc_ci_lo": r["macro_auc_ci"][0],
             "macro_auc_ci_hi": r["macro_auc_ci"][1],
             "seconds": r.get("seconds"),
             **{f"auc_{c}": r.get("per_class_auc", {}).get(c) for c in SNF_LABELS}}
            for i, r in enumerate(compared)
        ]).to_csv(OUT_DIR / "model_comparison.csv", index=False)
        _plot_compare_bars(compared, OUT_DIR / "model_comparison.png")

        best = next((r["name"] for r in compared if r["fit_ok"]), None)
        if best is None:
            raise RuntimeError("没有任何模型训练成功。")
        chosen_model = best
        print(f"\n👑 冠军模型: {chosen_model}")
    else:
        chosen_model = args.model
        print(f"使用指定模型: {chosen_model}")

    # ---------- 用最优/指定模型做最终训练 ----------
    print(f"\n最终训练 {chosen_model} 并生成产出 ...")
    cv = cross_validate_with_ci(
        X, y, numeric, categorical,
        n_splits=args.n_splits,
        random_state=args.random_state,
        n_boot=args.n_boot,
        n_estimators=args.n_estimators,
        model_name=chosen_model,
    )
    print(f"  Macro AUC = {cv.macro_auc:.3f}  95% CI [{cv.macro_auc_ci[0]:.3f}, {cv.macro_auc_ci[1]:.3f}]")
    for c in cv.labels:
        lo, hi = cv.per_class_auc_ci[c]
        print(f"    {c}: {cv.per_class_auc[c]:.3f}  [{lo:.3f}, {hi:.3f}]")

    pipe = build_pipeline(numeric, categorical,
                          n_estimators=args.n_estimators,
                          random_state=args.random_state,
                          model_name=chosen_model)
    pipe.fit(X, y)

    bundle_path = _save_bundle(pipe, cv, features, chosen_model)

    _plot_roc(cv.oof_true, cv.oof_proba, cv.labels,
              OUT_DIR / "roc_cv.png",
              f"5-fold CV ROC · {chosen_model}")
    _plot_confusion(cv.confusion_matrix, cv.labels, OUT_DIR / "confusion_matrix_cv.png")

    # 特征重要性(树模型) 或 系数绝对值(线性模型)
    try:
        names = pipe.named_steps["preproc"].get_feature_names_out()
    except Exception:
        names = None
    clf = pipe.named_steps["clf"]
    imp_values = None
    if hasattr(clf, "feature_importances_"):
        try: imp_values = clf.feature_importances_
        except Exception: pass
    elif hasattr(clf, "coef_"):
        coef = getattr(clf, "coef_")
        imp_values = np.abs(coef).mean(axis=0) if np.ndim(coef) == 2 else np.abs(coef)
    if names is not None and imp_values is not None:
        imp_df = pd.DataFrame({"feature": names, "importance": imp_values})
        imp_df = imp_df.sort_values("importance", ascending=False).head(20)
        imp_df.to_csv(OUT_DIR / "feature_importance_top20.csv", index=False)
        print("\nTop 10 important features:")
        print(imp_df.head(10).to_string(index=False))

    print("\n已保存:")
    for p in [
        bundle_path,
        OUT_DIR / "cv_metrics.json",
        OUT_DIR / "roc_cv.png",
        OUT_DIR / "confusion_matrix_cv.png",
        OUT_DIR / "feature_importance_top20.csv",
    ]:
        if p.exists():
            print(" -", p)
    if compared is not None:
        for p in [OUT_DIR / "model_comparison.csv", OUT_DIR / "model_comparison.png"]:
            if p.exists():
                print(" -", p)


if __name__ == "__main__":
    main()
