# 基于临床特征预测 SNF 4 分型 + 生存预测 + 相似病人比较

本仓库基于 Nature Genetics 2023 论文
[*Multi-omic stratification of luminal HR+/HER2- breast cancer*](https://www.nature.com/articles/s41588-023-01507-7)
公开的补充表(Table S1, N=579,其中 **351 例有 SNF1–4 标签**)。

原文用**数字病理 CNN** 或**转录组随机森林**来给出 SNF1–4 亚型,对没有病理切片 / 转录组数据的人来说,这两条路都走不通。

本仓库的目标是退而求其次:

> **只用医院常规病理报告里就能拿到的临床信息**(年龄、肿瘤大小、淋巴结、ER/PR%、Ki-67、分级、绝经、pT/pN、HER2 IHC、PAM50 如果做过)
> 训练一个分类器近似预测 SNF1–4,并在原文队列里找出和你最相似的病人来对照预后。

> ⚠️ 这是个**自学/科普性质**的工具,**不是医疗器械**,不能替代医生决策。
> 临床特征本身比转录组弱很多,Weighted AUC ≈ 0.71(原文转录组 ≈ 0.89,病理 CNN ≈ 0.81)。
> 仅适用于 **HR+(ER+ 或 PR+) / HER2-** 的乳腺癌。

---

## 一、数据与训练集

仓库里有论文公开的 4 个 Excel:

| 文件 | 说明 |
|---|---|
| `41588_2023_1507_MOESM3_ESM.xlsx` | **Supplementary Tables S1-S7(含 Table S1 临床 + SNF 标签,主要使用)** |
| `41588_2023_1507_MOESM5_ESM.xlsx` | Fig 3 source data |
| `41588_2023_1507_MOESM6_ESM.xlsx` | Fig 4 source data |
| `41588_2023_1507_MOESM12_ESM.xlsx` | Extended Data Fig 3(代谢组学等) |

我们只使用 `Table S1`。

### 训练数据 = 351 例有 SNF 标签的病人

Table S1 一共 579 行,其中 **228 行没有 SNF 标签**(可能原作者没做转录组聚类)。这部分会在训练时**自动剔除** —— 因为没有"正确答案",分类器学不了。

所以:
- **SNF 分类器的训练集 = 351 例**,分布如下:

  | Subtype | N | 简称 |
  |---|---:|---|
  | SNF1 | 86  | Canonical Luminal(经典 Luminal,预后较好) |
  | SNF2 | 89  | Immunogenic(免疫激活型) |
  | SNF3 | 118 | Proliferative(高增殖型) |
  | SNF4 | 58  | RTK-driven(RTK 驱动型,预后最差) |

- **不管你是 CLI、Web 前端还是静态 HTML 前端,给你做预测的都是同一个在这 351 例上训练的模型**。
- **相似病人**也是在这 351 例有标签 cohort 里找(无标签的病人连亚型都不知道,没法做对照)。
- **Cox 生存模型**则用更大的样本(OS 578 / RFS 575 / DMFS 566,有生存随访就行),只有包含 SNF 字段的 Cox 变体才会剔除到 ~350。详见章节六。

---

## 二、纯静态前端(零后端 · GitHub Pages 友好)

**给非技术用户最推荐的版本** —— 打开网页填表,浏览器本地推理,**不上传任何数据**。

```bash
# 1. 一次性把模型系数烤进 models.json
python3 src/export_static_models.py

# 2. 本地预览
cd static_app && python3 -m http.server 9876
# 浏览器打开 http://localhost:9876
```

**部署到 GitHub Pages / Netlify / Vercel / nginx 的详细步骤**:见 [`static_app/README.md`](static_app/README.md)。

仓库已经配置好 `.github/workflows/deploy-static.yml`:合并 `main` 后自动部署,第一次只需在 **Settings → Pages → Source = GitHub Actions** 开启一下。

特点:
- 渐变 hero + 3 步引导 UI(填表 → SNF 概率 → 4 种 SNF 假设的生存曲线 → 与原文对比)
- Chart.js 画图,移动端适配
- 前端纯 JS 复刻 sklearn/lifelines 推理:`softmax(coef·x + intercept)`、`S(t) = S₀(t)^exp(β·x)`,结果与后端**完全一致**
- 分类器用 **13 个核心临床字段**(不含辅助治疗),和下面的 CLI/FastAPI 默认口径一致

---

## 三、完整版(含 FastAPI 后端)

```bash
pip install -r requirements.txt
bash run_web.sh
# 浏览器打开 http://localhost:8000
```

Web 前端有 **6 个 Tab**:

1. **病人信息 / 预测** — 自动从队列生成表单(数值给出中位数提示,类别给出每个取值的样本数),支持"保存为默认 / 载入示例";预测结果显示每个亚型的**概率 + 95% 置信区间**和中文解读。
2. **训练配置 / 子人群** — 可以按任意字段过滤**训练集**(比如只用绝经、Grade 2–3、pN0–pN1、Ki67 ≥ 15%…),勾选参与训练的特征,**选择算法**,在该子人群上重新训练。
3. **模型大比拼** — 一次性训练 **16 种算法**(RandomForest / ExtraTrees / GradientBoosting / HistGradientBoosting / XGBoost / LightGBM / LogisticRegression / LogReg-L1 / LinearSVM / RBF-SVM / KNN / DecisionTree / GaussianNB / LDA / QDA / MLP)+ 6 个 **OvR 二分类组合**,按 **Weighted AUC** 排名,bootstrap 95% CI 误差线叠加原文 RF / CNN 参考线。勾选"自动采用最佳"后 Tab ① ⑤ 会换成冠军模型。
4. **模型评估 + 原文对比** — 当前模型详情:Weighted / Macro / Per-class AUC + bootstrap 95% CI、ROC、混淆矩阵、特征重要性、完整分类报告,与原文 Transcriptomics RF / Pathology CNN 并排。
5. **相似病人** — 可选"按特征重要性加权欧氏距离"、"只在预测亚型内找"。
6. **生存预测** — 为 **OS / RFS / DMFS** 各训 **4 个 Cox 变体**(2×2):
   - `base`(临床 only)
   - `+ Adjuvant therapy`
   - `+ SNF subtype`
   - `+ SNF + Adjuvant therapy`(默认显示)

   **两个 cohort 并排**(Full:各取最大样本;Matched:N=350 一致公平对比)。含 SNF 的 Cox 变体支持**"把病人依次假设成 SNF1/2/3/4"** 四条曲线对比 + Expected 按概率加权虚线。

> **训练/预测是会话内的**:Tab ② 训练出的模型会自动用于 Tab ① / ⑤;重启服务会回到默认全队列模型。

### API 路由

| 路由 | 方法 | 说明 |
|---|---|---|
| `/` | GET | 主页 |
| `/api/meta` | GET | 字段范围 / 类别分布 / 可用算法 |
| `/api/benchmarks` | GET | 原文 Transcriptomics RF + Pathology CNN AUC |
| `/api/train` | POST | 选特征 + 子人群 + 算法训练 |
| `/api/compare` | POST | 22 模型大比拼,返回排行榜,可自动采用最佳 |
| `/api/predict` | POST | 预测一个病人 |
| `/api/similar` | POST | 找最相似 Top-K |
| `/api/survival/train` | POST | 训练 OS/RFS/DMFS 各 4 变体,返回 train/test/CV C-index(Full + Matched 两组 cohort) |
| `/api/survival/predict` | POST | 个人生存曲线 + 4 种 SNF 假设曲线 + Expected |
| `/api/survival/status` | GET | 当前已训练哪些端点 |

---

## 四、命令行使用

### 1. 复制 / 编辑病人模板

```bash
cp patient_template.yaml my_patient.yaml
# 编辑,把所有字段改成你自己的
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
HER2_IHC_Status: 1   # HER2 IHC: 0 / 1 / 2(HER2- 的话一般是 0/1/2 且 FISH-)

Menopause: "No"      # "Yes" / "No"
Grade: 2             # 1 / 2 / 3
pT: "pT2"            # "pT1" / "pT2" / "pT3"
pN: "pN0"            # "pN0" / "pN1" / "pN2" / "pN3"
PR_status: "Positive"
PAM50: null          # 没做过就保持 null

# 术后辅助治疗(术前病人请保持 null)
Adjuvant_chemotherapy: null       # "Yes" / "No"
Adjuvant_radiotherapy: null
Adjuvant_endocrine_therapy: null
```

任何缺失字段写 `null` 即可,模型会自动填补(数值用中位数,类别用 `"Missing"`)。

### 2. 一键跑全流程

```bash
bash run_all.sh my_patient.yaml
```

依次做:

1. **模型大比拼 + 自动选最佳**(`src/model.py`):16 + 6 OvR 个算法按 weighted AUC 排名,保存冠军到 `outputs/snf_classifier.pkl`,产出 `model_comparison.csv/png`、`cv_metrics.json`、`roc_cv.png`、`confusion_matrix_cv.png`、`feature_importance_top20.csv`。
2. **预测 SNF 亚型**(`src/predict_patient.py`):`outputs/prediction_ME.json`,控制台打印每类概率 + 95% CI + 使用的模型名。
3. **找 Top-20 相似病人**(`src/find_similar.py`):按特征重要性加权欧氏距离,`outputs/similar_patients_ME.csv`。
4. **画 KM 曲线**(`src/survival_compare.py`)。
5. **CoxPH 个人生存预测**(`src/survival_predict.py`):OS/RFS/DMFS × 4 变体 × (Full + Matched cohort),产出 `survival_report_ME.csv` / `..._matched.csv` / `survival_prediction_ME.json` / `survival_curve_ME.png` / `survival_curve_ME_bySNF.png`(4 种 SNF 假设图)。

### 关于辅助治疗默认值

| 场景 | 辅助治疗 3 字段默认 |
|---|---|
| CLI `bash run_all.sh`(SNF 分型部分) | **不含** |
| Web 前端 Tab ①②(SNF 分型) | **不含**(`Adjuvant_*` 复选框初始未勾选) |
| 静态 HTML 前端(SNF 分型) | **不含**(烤进 `models.json` 的分类器就是 13 字段) |
| 生存预测 `src/survival_predict.py` | **含**(术后病人治疗确实影响生存,这里保留) |

**如果你是术前病人**(还没决定要不要化疗),三个字段在 YAML 里保持 `null` 即可,不会影响分型。生存预测里 Cox 模型会自动把 `Missing` 当一个类别处理。

### 常见命令变体

```bash
# 强制使用指定算法(跳过大比拼, 节省时间)
bash run_all.sh my_patient.yaml --model RandomForest
MODEL=LogReg-L1 bash run_all.sh my_patient.yaml

# 让 SNF 分型也把辅助治疗当特征(仅建议术后病人使用)
bash run_all.sh my_patient.yaml --with-treatment
WITH_TREATMENT=1 bash run_all.sh my_patient.yaml
```

### 3. 单独跑某一步

```bash
python3 src/model.py                     # 只训练(默认大比拼+自动选)
python3 src/model.py --list-models       # 看有哪些算法
python3 src/model.py --model LogReg-L1   # 指定算法
python3 src/model.py --compare RandomForest XGBoost LogReg-L1 LinearSVM LDA

python3 src/predict_patient.py --patient my_patient.yaml

python3 src/find_similar.py --patient my_patient.yaml --k 15
python3 src/find_similar.py --patient my_patient.yaml --k 15 --same-subtype-only
python3 src/find_similar.py --patient my_patient.yaml --k 15 --no-weight

python3 src/survival_compare.py --patient my_patient.yaml --k 20
python3 src/survival_predict.py --patient my_patient.yaml
```

---

## 五、模型说明

### SNF 分型

- **训练集**:Table S1 里 **351 例有 SNF 标签的 HR+/HER2- 病人**(无标签的 228 例不参与训练)
- **特征**:7 个数值 + 6 个类别 = **13 列**(不含辅助治疗)
  - 数值:Age / Tumor_size_cm / Positive_axillary_lymph_nodes / ER_percent / PR_percent / Ki67 / HER2_IHC_Status
  - 类别:Menopause / Grade / pT / pN / PR_status / PAM50
- **预处理**:数值中位数填补 + z-score 标准化;类别 `Missing` 填补 + One-Hot
- **算法**:**大比拼后自动选最佳**。在这份数据上通常是 **LogReg-L1** 或 LinearSVM/LDA 夺冠(低维小样本 + 线性信号主导),RandomForest 排第 5 左右。
- **评估**:5 折 Stratified CV + **bootstrap 95% CI**,同时报告 **Weighted AUC**(推荐主用,按类别样本数加权)和 Macro AUC。

当前默认数据上的表现(seed=42):

| Subtype | 本工具 CV AUC | 原文 Transcriptomics RF | 原文 Pathology CNN |
|---|---:|---:|---:|
| SNF1 | 0.78 | 0.95 | 0.87 |
| SNF2 | 0.62 | 0.93 | 0.81 |
| SNF3 | 0.75 | 0.85 | 0.78 |
| SNF4 | 0.68 | 0.82 | 0.78 |
| **Weighted** | **~0.71** | 0.89 | 0.81 |

**结论**:只靠临床信息,SNF1(经典 Luminal)最容易识别,SNF2(免疫型)最难 —— 因为 SNF2 的判别信号主要在免疫细胞浸润 / 转录组里,临床无对应字段。**这个模型适合做"我大概率不是某些亚型"的筛查,不要当确诊工具。**

### 相似病人

1. 把你和队列里 351 个病人一起过**训练时的预处理管道**,得到约 30 维的特征空间
2. 默认**按 RandomForest/L1 LR 的特征重要性加权**,让判别力强的维度(年龄 / Ki67 / PR% / PAM50)主导距离
3. 欧氏距离升序取 Top-K
4. 可选 `--same-subtype-only` / `--no-weight`

注意:**相似 ≠ 同亚型**,Top-K 里混着其他亚型是正常的。

---

## 六、生存预测(CoxPH)

对 OS / RFS / DMFS 三个端点,分别训练 **4 种 Cox 变体**:

| 变体 | 含 SNF? | 含辅助治疗? |
|---|:---:|:---:|
| `base` | ✗ | ✗ |
| `treat` | ✗ | ✓ |
| `snf` | ✓ | ✗ |
| `snf+treat`(默认) | ✓ | ✓ |

每个变体同时在两个 cohort 上训练:

| Cohort | N | 用途 |
|---|---:|---|
| **Full** | ~578(base/treat)/ ~350(含 SNF) | 各取最大样本,反映最佳性能 |
| **Matched** | **~350(所有 4 变体一致)** | 公平对比:加 SNF/治疗到底提升多少 |

实测在 Matched cohort 上(同一 350 例):

| Endpoint | base | + treat | + snf | + snf+treat |
|---|---:|---:|---:|---:|
| OS CV C-index | 0.730 | 0.724 | **0.743** | 0.735 |
| RFS | 0.724 | 0.729 | 0.727 | **0.731** |
| DMFS | 0.736 | 0.742 | 0.749 | **0.752** |

对个人:**同一病人在 4 种 SNF 假设下 10 年生存概率可以差 20–25 个百分点** —— 即使群体 C-index 只涨 0.01。两者不矛盾,前端 Tab ⑥ 有详细 explainer。

---

## 七、文件结构

```
.
├── 41588_2023_1507_MOESM*.xlsx     # 原文 4 个 Excel
├── patient_template.yaml           # 病人信息模板
├── requirements.txt
├── run_all.sh                      # CLI 一键脚本
├── run_web.sh                      # 启动 FastAPI + 原生 HTML 前端
├── README.md                       # 本文件
├── src/
│   ├── data_loader.py              # 读 Table S1 + 清洗
│   ├── training.py                 # 训练 + 22 算法 + 子人群 + bootstrap CI
│   ├── survival.py                 # CoxPH + 4 变体 + 4 SNF 假设 + Matched cohort
│   ├── model.py                    # CLI: 大比拼 + 自动选最佳
│   ├── predict_patient.py          # CLI: 预测一人
│   ├── find_similar.py             # CLI: 相似病人
│   ├── survival_compare.py         # CLI: KM 曲线
│   ├── survival_predict.py         # CLI: CoxPH 个人生存
│   └── export_static_models.py     # 导出系数到 static_app/models.json
├── web/
│   ├── app.py                      # FastAPI 后端
│   └── static/                     # 原生 HTML/CSS/JS 前端(6 Tab)
├── static_app/
│   ├── index.html, style.css, app.js   # 零后端静态前端
│   ├── models.json                 # 烤好的模型系数(~370 KB)
│   └── README.md                   # 部署指南
├── .github/workflows/deploy-static.yml   # GH Pages 自动部署
└── outputs/                        # CLI 产物
```

---

## 八、引用

如果用到了原数据,请引用:

> Gong Y, Ji P, Yang YS, et al. *Multi-omic stratification of HR+/HER2- breast cancer reveals integrative subtypes with prognostic and therapeutic relevance.*
> **Nature Genetics** 55, 1716–1730 (2023). https://doi.org/10.1038/s41588-023-01507-7
