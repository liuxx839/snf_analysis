# 基于临床特征预测 SNF 4 分型 + 相似病人比较

本仓库基于 Nature Genetics 2023 论文
[*Multi-omic stratification of luminal HR+/HER2- breast cancer*](https://www.nature.com/articles/s41588-023-01507-7)
公开的补充表(Table S1, N=579, 其中 351 例有 SNF1–4 标签)。

原文是用**数字病理 CNN** 或 **转录组随机森林**来给出 SNF1–4 亚型。
对没有病理切片 / 转录组数据的人来说,这两条路都走不通。

本仓库的目标是退而求其次:
**只用医院常规病理报告里就能拿到的临床信息**(年龄、肿瘤大小、淋巴结、ER/PR%、Ki-67、分级、绝经、pT/pN、HER2 IHC、PAM50 如果做过) 训练一个分类器,
近似预测 SNF1–4,并在原文队列里找出和你最相似的病人来对照预后。

> ⚠️ 这是个**自学/科普性质**的工具,不是医疗器械,不能替代医生决策。
> 临床特征本身比转录组弱很多,Macro AUC 只在 ~0.69 量级,**远低于**原文转录组 (~0.89) 或病理 CNN (~0.81) 的水平。
> 仅适用于 HR+(ER+ 或 PR+) / HER2- 的乳腺癌。

---

## 一、数据

仓库里有论文公开的 4 个 Excel:

| 文件 | 说明 |
|---|---|
| `41588_2023_1507_MOESM3_ESM.xlsx` | Supplementary Tables S1-S7 (含 Table S1 临床+SNF 标签) |
| `41588_2023_1507_MOESM5_ESM.xlsx` | Fig 3 source data |
| `41588_2023_1507_MOESM6_ESM.xlsx` | Fig 4 source data |
| `41588_2023_1507_MOESM12_ESM.xlsx` | Extended Data Fig 3 (代谢-组学等) |

我们只使用 `Table S1`。可用临床字段(以及缺失率)见 `python3 src/data_loader.py` 输出。

SNF 亚型分布(351 例):

| Subtype | N | 简称 |
|---|---:|---|
| SNF1 | 86  | Canonical Luminal(经典 Luminal,预后较好) |
| SNF2 | 89  | Immunogenic(免疫激活) |
| SNF3 | 118 | Proliferative(增殖型) |
| SNF4 | 58  | RTK-driven(RTK 驱动,预后最差) |

---

## 二、安装

```bash
pip install -r requirements.txt
```

只用 CPU,几秒内能跑完。

---

## 三、使用

### 1. 复制 / 编辑你自己的病人模板

```bash
cp patient_template.yaml my_patient.yaml
# 用编辑器把里面的字段改成你自己的
```

模板字段:

```yaml
patient_id: "ME"

Age: 45
Tumor_size_cm: 2.2
Positive_axillary_lymph_nodes: 0

ER_percent: 90
PR_percent: 80
Ki67: 20
HER2_IHC_Status: 1   # HER2 IHC 评分: 0 / 1 / 2

Menopause: "No"      # "Yes" / "No"
Grade: 2             # 1 / 2 / 3
pT: "pT2"            # "pT1" / "pT2" / "pT3"
pN: "pN0"            # "pN0" / "pN1" / "pN2" / "pN3"
PR_status: "Positive"  # "Positive" / "Negative"
PAM50: null          # 没做过就保持 null
```

任何缺失字段写 `null` 即可,模型会自动填补(数值用中位数,类别用"Missing")。

### 2. 一键跑全流程

```bash
bash run_all.sh my_patient.yaml
```

它会按顺序做:

1. **训练 + 5 折交叉验证** (`src/model.py`):产出 `outputs/snf_classifier.pkl`,以及
   `outputs/roc_cv.png`、`outputs/confusion_matrix_cv.png`、`outputs/cv_metrics.json`、
   `outputs/feature_importance_top20.csv`。
2. **预测你的 SNF 亚型** (`src/predict_patient.py`):产出 `outputs/prediction_ME.json`,
   控制台直接打印每个亚型的概率以及 5 折 CV AUC。
3. **找最相似的 Top-20 病人** (`src/find_similar.py`):产出 `outputs/similar_patients_ME.csv`。
4. **画 KM 曲线** (`src/survival_compare.py`):
   - `outputs/km_OS_by_SNF.png` / `km_RFS_by_SNF.png` / `km_DMFS_by_SNF.png` —— 4 个亚型在原队列里的生存曲线对比;
   - `outputs/km_OS_similar_ME.png` 等 —— "你的相似病人 vs 全队列"的生存对比。

### 3. 单独跑某一步

```bash
python3 src/model.py                                       # 训练+评估
python3 src/predict_patient.py --patient my_patient.yaml   # 预测
python3 src/find_similar.py --patient my_patient.yaml --k 15
python3 src/find_similar.py --patient my_patient.yaml --k 15 --same-subtype-only  # 只在同亚型内找
python3 src/survival_compare.py --patient my_patient.yaml --k 20
```

---

## 四、模型说明

- **算法**:`RandomForest`(800 棵, `class_weight='balanced'`)。原文 transcriptomics 模型也是随机森林。
- **特征**:7 个数值 + 6 个类别,共 13 列,经过 `ColumnTransformer`:
  - 数值:中位数填补 + 标准化
  - 类别:常数填补 + One-Hot
- **评估**:分层 5 折交叉验证, one-vs-rest AUC(原文也是这种报告方式)。

在仓库默认数据上跑出来的指标(随机种子 42):

| Subtype | CV AUC (临床特征模型) | 原文 transcriptomics RF | 原文 pathology CNN |
|---|---:|---:|---:|
| SNF1 | ~0.79 | 0.95 | 0.87 |
| SNF2 | ~0.58 | 0.93 | 0.81 |
| SNF3 | ~0.73 | 0.85 | 0.78 |
| SNF4 | ~0.66 | 0.82 | 0.78 |
| Macro | **~0.69** | 0.89 | 0.81 |

**结论**:只靠临床信息,SNF1(经典 Luminal)最容易识别,SNF2(免疫亚型)最难,
因为 SNF2 的判别信号主要在免疫细胞浸润 / 转录组里,临床无对应字段。
这个模型适合作为"我大概率不是某些亚型"的过滤器,**不要把它当作确诊工具**。

---

## 五、相似病人

`find_similar.py` 在 `RandomForest` 预处理后的特征空间里(标准化数值 + One-Hot 类别)
计算欧氏距离,返回最近的 K 个病人,以及他们的 SNF 标签、PAM50、肿瘤参数和 OS/RFS/DMFS。
随后 `survival_compare.py` 把这部分人和整个队列做 KM 曲线对比。

注意:相似 ≠ 同亚型。如果你只想看同亚型内的相似病人,加 `--same-subtype-only`。

---

## 六、文件结构

```
.
├── 41588_2023_1507_MOESM3_ESM.xlsx     # Table S1 (主要使用)
├── 41588_2023_1507_MOESM5_ESM.xlsx
├── 41588_2023_1507_MOESM6_ESM.xlsx
├── 41588_2023_1507_MOESM12_ESM.xlsx
├── patient_template.yaml               # 病人信息模板
├── requirements.txt
├── run_all.sh                          # 一键脚本
├── README.md
├── src/
│   ├── data_loader.py                  # 读 Table S1 + 类型清洗
│   ├── model.py                        # 训练 + 5 折 CV + 保存模型
│   ├── predict_patient.py              # 用模型预测一个病人
│   ├── find_similar.py                 # 找最相似 Top-K 病人
│   └── survival_compare.py             # KM 生存曲线对比
└── outputs/                            # 所有结果(自动生成)
```

---

## 七、引用

如果用到了原数据,请引用:

> Gong Y, Ji P, Yang YS, et al. *Multi-omic stratification of HR+/HER2- breast cancer reveals
> integrative subtypes with prognostic and therapeutic relevance.* **Nature Genetics** 55, 1716–1730 (2023).
> https://doi.org/10.1038/s41588-023-01507-7
