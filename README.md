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

## 三、Web 前端(推荐)

```bash
bash run_web.sh
# 然后浏览器打开 http://localhost:8000
```

前端基于 FastAPI + 原生 HTML/JS(不依赖 Streamlit),有 5 个 Tab:

1. **病人信息 / 预测** — 自动从队列生成表单(数值给出中位数提示,类别给出每个取值的样本数),
   支持"保存为默认 / 载入默认 / 载入示例";预测结果显示每个亚型的**概率 + 95% 置信区间**
   和自动生成的中文解读。对 RF/ExtraTrees 这类 bagging 模型,CI 用"树级方差"给出;
   其它模型退化为点估计。
2. **训练配置 / 子人群** — 可以按任意字段过滤**训练集**(比如只用绝经、Grade 2–3、pN0–pN1、Ki67 ≥ 15% ...),
   勾选参与训练的特征,**选择算法**(16 个之一),调整 CV 折数 / Bootstrap 次数 / 树数,
   然后在该子人群上重新训练。
3. **模型大比拼** — 一次性同时训练 **16 种主流算法**(RandomForest / ExtraTrees / GradientBoosting /
   HistGradientBoosting / XGBoost / LightGBM / LogisticRegression / LogReg-L1 / LinearSVM / RBF-SVM /
   KNN / DecisionTree / GaussianNB / LDA / QDA / MLP),按 Macro AUC 排名,还会画出每个模型的柱状图 +
   bootstrap 95% CI 误差线,并把原文 Transcriptomics RF / Pathology CNN 作为参考线叠加。
   可选"**自动采用最佳模型**",这样 Tab ①④ 的预测/相似病人都会换成表现最好的算法。
4. **模型评估 + 原文对比** — 上一次训练模型的详情:Macro / Per-class AUC + bootstrap 95% CI、
   ROC、混淆矩阵、特征重要性、完整分类报告,一张表对比原文 Transcriptomics RF 和 Pathology CNN。
5. **相似病人** — 可以选择"是否按特征重要性加权欧氏距离"、"是否只在预测亚型内找"。

> **注:辅助治疗三字段** (`Adjuvant_chemotherapy` / `Adjuvant_radiotherapy` / `Adjuvant_endocrine_therapy`)
> 是"治疗端"变量,不是肿瘤本身的特性。对术前病人无法获得,并可能间接泄露医生看报告后做出的决策,
> 所以默认**不参与训练**,表单里有特殊标记;勾选后视作术后病人的额外信息。

> 训练/预测都是**会话内**的:你在 Tab ② 训练出的模型会自动用于 Tab ① 的预测和 Tab ④ 的相似度计算,
> 重启服务会回退到默认全队列模型。

### API 路由

| 路由 | 方法 | 说明 |
|---|---|---|
| `/` | GET | 主页 |
| `/api/meta` | GET | 队列字段范围 / 类别取值分布 |
| `/api/benchmarks` | GET | 原文 Transcriptomics RF + Pathology CNN AUC |
| `/api/train` | POST | 选特征+子人群+算法训练, 返回 AUC/CI/ROC/混淆矩阵/特征重要性 |
| `/api/compare` | POST | 一次性比较多个模型, 返回排行榜, 可选择自动采用最佳 |
| `/api/predict` | POST | 预测一个病人的 SNF 亚型 + 每类概率 CI |
| `/api/similar` | POST | 找最相似 Top-K 病人 |

---

## 四、命令行使用

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

1. **模型大比拼 + 自动选最佳** (`src/model.py`):跑 16 个算法(RandomForest / XGBoost / LightGBM /
   LogisticRegression / LogReg-L1 / LinearSVM / LDA / … / MLP),按 Macro AUC 排名并自动采用冠军,
   产出 `outputs/snf_classifier.pkl`、`outputs/model_comparison.csv/png`、`outputs/cv_metrics.json`、
   `outputs/roc_cv.png`、`outputs/confusion_matrix_cv.png`、`outputs/feature_importance_top20.csv`。
2. **预测你的 SNF 亚型** (`src/predict_patient.py`):产出 `outputs/prediction_ME.json`,
   控制台直接打印每个亚型的概率 + CV AUC + 95% CI,并显示当前用的是哪个模型。
3. **找最相似的 Top-20 病人** (`src/find_similar.py`):按特征重要性加权的欧氏距离,
   产出 `outputs/similar_patients_ME.csv`。
4. **画 KM 曲线** (`src/survival_compare.py`)。

常见变体:

```bash
# 强制使用指定算法(跳过大比拼, 节省时间)
bash run_all.sh my_patient.yaml --model RandomForest
MODEL=LogReg-L1 bash run_all.sh my_patient.yaml

# 把术后辅助治疗三字段也作为特征(仅限术后病人)
bash run_all.sh my_patient.yaml --with-treatment
WITH_TREATMENT=1 bash run_all.sh my_patient.yaml
```

和 **Web 前端完全一致**:都共享 `src/training.py`,都能做 16 模型比拼 + bootstrap 95% CI 评估。
CLI 默认"自动选最佳",Web 前端默认 RandomForest(点击 Tab ③"模型大比拼"可以切换)。

### 3. 单独跑某一步

```bash
# 只训练, 默认跑大比拼并自动选最佳
python3 src/model.py

# 看有哪些算法可选
python3 src/model.py --list-models

# 指定算法
python3 src/model.py --model LogReg-L1

# 只对比几个模型
python3 src/model.py --compare RandomForest XGBoost LogReg-L1 LinearSVM LDA

# 预测一个病人(自动加载 outputs/snf_classifier.pkl)
python3 src/predict_patient.py --patient my_patient.yaml

# 找相似病人(按特征重要性加权,是默认)
python3 src/find_similar.py --patient my_patient.yaml --k 15
python3 src/find_similar.py --patient my_patient.yaml --k 15 --same-subtype-only
python3 src/find_similar.py --patient my_patient.yaml --k 15 --no-weight

python3 src/survival_compare.py --patient my_patient.yaml --k 20
```

---

## 五、模型说明

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

## 六、相似病人

**相似度的定义**:
1. 把你和队列里每个病人的临床特征一起走**训练时的预处理管道**
   (数值 → 中位数填补 + z-score 标准化;类别 → 缺失填 `Missing` + One-Hot),
   得到同一个约 30 维特征空间。
2. 默认计算**欧氏距离** `d = ||x_me − x_patient||₂`,按 `d` 升序取 Top-K。
3. Web 前端 / API 支持"**按特征重要性加权**":把每维按 RandomForest 训练出的
   `feature_importances_` 归一后,乘进距离里,让更能判别 SNF 的维度说了算
   (比如年龄 / Ki67 / PR% / PAM50 权重大,肿瘤大小权重小)。
4. 也支持"只在预测出的同亚型内找"(`--same-subtype-only` / 前端勾选框)。

注意:相似 ≠ 同亚型。只按临床相似,返回的 Top-K 里仍可能混着其他亚型,这是正常的。

---

## 七、文件结构

```
.
├── 41588_2023_1507_MOESM3_ESM.xlsx     # Table S1 (主要使用)
├── 41588_2023_1507_MOESM5_ESM.xlsx
├── 41588_2023_1507_MOESM6_ESM.xlsx
├── 41588_2023_1507_MOESM12_ESM.xlsx
├── patient_template.yaml               # 病人信息模板
├── requirements.txt
├── run_all.sh                          # CLI 一键脚本
├── run_web.sh                          # 启动 Web 前端
├── README.md
├── src/
│   ├── data_loader.py                  # 读 Table S1 + 类型清洗
│   ├── training.py                     # 可复用训练/评估(子人群 + bootstrap CI + 森林 CI)
│   ├── model.py                        # CLI: 训练 + 5 折 CV + 保存模型
│   ├── predict_patient.py              # CLI: 预测一个病人
│   ├── find_similar.py                 # CLI: 找最相似 Top-K
│   └── survival_compare.py             # CLI: KM 生存曲线
├── web/
│   ├── app.py                          # FastAPI 后端
│   └── static/                         # 原生 HTML/CSS/JS 前端
└── outputs/                            # CLI 产物
```

---

## 八、引用

如果用到了原数据,请引用:

> Gong Y, Ji P, Yang YS, et al. *Multi-omic stratification of HR+/HER2- breast cancer reveals
> integrative subtypes with prognostic and therapeutic relevance.* **Nature Genetics** 55, 1716–1730 (2023).
> https://doi.org/10.1038/s41588-023-01507-7
