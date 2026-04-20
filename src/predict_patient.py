"""
用训练好的模型, 对你自己的信息做 SNF 预测, 并给出解释。
用法:
    python src/predict_patient.py --patient patient_template.yaml
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from data_loader import ALL_FEATURES, SNF_LABELS
from data_loader import SNF_DESCRIPTION

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
MODEL_PATH = OUT_DIR / "snf_classifier.pkl"


def load_patient(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data


def patient_to_row(patient: dict) -> pd.DataFrame:
    row = {feat: patient.get(feat, None) for feat in ALL_FEATURES}
    df = pd.DataFrame([row])
    for c in ["Age", "Tumor_size_cm", "Positive_axillary_lymph_nodes",
              "ER_percent", "PR_percent", "Ki67", "HER2_IHC_Status"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["Menopause", "Grade", "pT", "pN", "PR_status", "PAM50"]:
        if c in df.columns:
            df[c] = df[c].astype("object").where(df[c].notna(), None)
            df[c] = df[c].apply(lambda v: None if v is None else str(v))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", type=str, default="patient_template.yaml",
                    help="病人信息 YAML 文件")
    ap.add_argument("--model", type=str, default=str(MODEL_PATH))
    args = ap.parse_args()

    patient = load_patient(Path(args.patient))
    patient_id = patient.get("patient_id", "ME")

    with open(args.model, "rb") as f:
        bundle = pickle.load(f)
    pipe = bundle["pipeline"]
    labels = bundle["labels"]
    cv_metrics = bundle.get("cv_metrics", {})

    X_new = patient_to_row(patient)
    proba = pipe.predict_proba(X_new)[0]
    col_idx = {c: list(pipe.classes_).index(c) for c in labels}
    proba_sorted = np.array([proba[col_idx[c]] for c in labels])
    pred = labels[int(np.argmax(proba_sorted))]

    print("=" * 60)
    print(f"病人: {patient_id}")
    print("=" * 60)
    print("\n输入特征:")
    for k in ALL_FEATURES:
        print(f"  {k:35s}: {patient.get(k, '(缺失, 模型会自动填补)')}")

    print("\n---- SNF 分型预测 ----")
    for c in labels:
        auc = cv_metrics.get("per_class_auc", {}).get(c, None)
        auc_str = f"(CV AUC = {auc:.3f})" if auc is not None else ""
        print(f"  P({c}) = {proba_sorted[labels.index(c)]:.3f}   {auc_str}")
    print(f"\n最可能的 SNF 亚型: {pred}")
    print(f"  {SNF_DESCRIPTION.get(pred, '')}")

    print("\n---- 模型整体性能(5 折 CV) ----")
    print(f"  Macro AUC = {cv_metrics.get('macro_auc', float('nan')):.3f}")
    for c in labels:
        a = cv_metrics.get("per_class_auc", {}).get(c, None)
        if a is not None:
            print(f"  {c} AUC = {a:.3f}")

    OUT_DIR.mkdir(exist_ok=True)
    out = {
        "patient_id": patient_id,
        "input": {k: patient.get(k, None) for k in ALL_FEATURES},
        "predicted_subtype": pred,
        "probabilities": {c: float(proba_sorted[labels.index(c)]) for c in labels},
        "subtype_description": SNF_DESCRIPTION.get(pred, ""),
        "model_cv_metrics": cv_metrics,
    }
    out_path = OUT_DIR / f"prediction_{patient_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {out_path}")


if __name__ == "__main__":
    main()
