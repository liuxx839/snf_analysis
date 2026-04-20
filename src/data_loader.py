"""
数据加载与清洗
==============

从 Nature 2023 (s41588-023-01507-7) 补充材料 Table S1 中读取临床数据,
做最小程度的特征工程,产出建模可直接使用的 X / y 以及原始信息。

只使用临床特征(年龄、肿瘤大小、阳性淋巴结、ER/PR%、Ki67、分级、绝经、pT/pN、HER2 IHC、PAM50),
不依赖病理图像或转录组,因此只要有医院常规的免疫组化和病理报告即可。
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd

DEFAULT_S1 = Path(__file__).resolve().parent.parent / "41588_2023_1507_MOESM3_ESM.xlsx"

NUMERIC_FEATURES: List[str] = [
    "Age",
    "Tumor_size_cm",
    "Positive_axillary_lymph_nodes",
    "ER_percent",
    "PR_percent",
    "Ki67",
    "HER2_IHC_Status",
]

CATEGORICAL_FEATURES: List[str] = [
    "Menopause",
    "Grade",
    "pT",
    "pN",
    "PR_status",
    "PAM50",
]

# 术后辅助治疗(adjuvant therapy) —— "手术之后额外加上"的治疗:
#   - Adjuvant_chemotherapy      : 辅助化疗
#   - Adjuvant_radiotherapy      : 辅助放疗
#   - Adjuvant_endocrine_therapy : 辅助内分泌治疗(他莫昔芬/AI 等)
# 这些是"治疗端"变量,而不是"肿瘤本身"的特性; 把它们当特征预测 SNF 时
# 需要注意:1) 医生在选治疗时可能已经间接知道了肿瘤风险, 会造成轻微的信息泄漏;
# 2) 对"术前"病人不可用。因此默认放入特征池, 但前端会提供开关让用户按需选入。
TREATMENT_FEATURES: List[str] = [
    "Adjuvant_chemotherapy",
    "Adjuvant_radiotherapy",
    "Adjuvant_endocrine_therapy",
]

# 所有可用临床特征 = 核心临床 + 辅助治疗
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + TREATMENT_FEATURES

FEATURE_DESCRIPTION: dict = {
    "Age": "年龄(岁)",
    "Tumor_size_cm": "肿瘤最大径(cm)",
    "Positive_axillary_lymph_nodes": "阳性腋窝淋巴结数",
    "ER_percent": "ER 免疫组化阳性比例(0-100%)",
    "PR_percent": "PR 免疫组化阳性比例(0-100%)",
    "Ki67": "Ki-67 指数(%)",
    "HER2_IHC_Status": "HER2 IHC 评分 0/1/2/3",
    "Menopause": "是否绝经",
    "Grade": "病理分级 1/2/3",
    "pT": "术后 T 分期",
    "pN": "术后 N 分期(淋巴结)",
    "PR_status": "PR 阳性/阴性",
    "PAM50": "PAM50 分子分型(LumA/LumB/Her2/Basal/Normal)",
    "Adjuvant_chemotherapy":     "术后辅助化疗(Yes/No) —— 治疗端变量; 术后病人建议填入, 术前病人请留空",
    "Adjuvant_radiotherapy":     "术后辅助放疗(Yes/No) —— 治疗端变量; 术后病人建议填入, 术前病人请留空",
    "Adjuvant_endocrine_therapy":"术后辅助内分泌治疗(Yes/No) —— 治疗端变量; 术后病人建议填入, 术前病人请留空",
}

SNF_LABELS = ["SNF1", "SNF2", "SNF3", "SNF4"]
SNF_DESCRIPTION = {
    "SNF1": "Canonical Luminal(经典 Luminal):基因组相对稳定,内分泌治疗反应较好,预后总体较好。",
    "SNF2": "Immunogenic(免疫激活):免疫细胞浸润高,可能从免疫治疗联合中获益,预后中等偏好。",
    "SNF3": "Proliferative(增殖型):增殖通路活跃、Ki67 偏高、CDK4/6 通路相关,常需更强化疗或 CDK4/6 抑制剂。",
    "SNF4": "RTK-driven(受体酪氨酸激酶驱动):预后最差,该亚型中 RTK/HER 家族信号活跃,可能需要靶向 RTK/PARP 联合方案。",
}


def _to_float(value):
    """把 'Unknown' / 字符串数字 / NaN 统一变成 float 或 np.nan。"""
    if value is None:
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        f = float(value)
        return f if not np.isnan(f) else np.nan
    s = str(value).strip()
    if s == "" or s.lower() in {"unknown", "nan", "na", "none", "n/a"}:
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def _to_str(value):
    if value is None:
        return np.nan
    if isinstance(value, float) and np.isnan(value):
        return np.nan
    s = str(value).strip()
    if s == "" or s.lower() in {"unknown", "nan", "na", "none", "n/a"}:
        return np.nan
    return s


def load_table_s1(path: str | Path = DEFAULT_S1) -> pd.DataFrame:
    """读取 Table S1 原始表,header 在第 2 行(索引 1)。"""
    df = pd.read_excel(path, sheet_name="Table S1", header=1)
    return df


def build_feature_frame(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    清洗 + 类型转换。
    返回 DataFrame,索引为 PatientCode,包含所有 ALL_FEATURES 列、SNF_subtype 以及生存信息。
    """
    df = df_raw.copy()
    df = df.set_index("PatientCode")

    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = df[col].map(_to_float)

    for col in CATEGORICAL_FEATURES + TREATMENT_FEATURES:
        if col in df.columns:
            df[col] = df[col].map(_to_str)

    df["SNF_subtype"] = df["SNF_subtype"].map(_to_str)

    surv_cols = [
        "OS_status", "OS_months",
        "RFS_status", "RFS_months",
        "DMFS_status", "DMFS_months",
    ]
    for c in surv_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def split_labeled_unlabeled(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    labeled = df[df["SNF_subtype"].isin(SNF_LABELS)].copy()
    unlabeled = df[~df["SNF_subtype"].isin(SNF_LABELS)].copy()
    return labeled, unlabeled


def get_modeling_matrix(
    df: pd.DataFrame, features: List[str] | None = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    返回特征 DataFrame X(还未编码)和目标 Series y。
    """
    if features is None:
        features = ALL_FEATURES
    available = [c for c in features if c in df.columns]
    X = df[available].copy()
    y = df["SNF_subtype"].copy()
    return X, y


if __name__ == "__main__":
    raw = load_table_s1()
    feats = build_feature_frame(raw)
    labeled, unlabeled = split_labeled_unlabeled(feats)
    print(f"总样本: {len(feats)}, 有 SNF 标签: {len(labeled)}, 无标签: {len(unlabeled)}")
    print("SNF 分布:")
    print(labeled["SNF_subtype"].value_counts())
    X, y = get_modeling_matrix(labeled)
    print("特征矩阵 shape:", X.shape)
    print("缺失比例:")
    print((X.isna().mean() * 100).round(1))
