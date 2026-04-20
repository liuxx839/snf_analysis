#!/usr/bin/env bash
# 一键: 模型大比拼(自动选最佳) -> 预测病人 -> 找相似病人 -> 画 KM 曲线
#
# 用法:
#   bash run_all.sh [patient_yaml] [--model MODEL_NAME] [--with-treatment]
#
# 默认不加术后辅助治疗三字段(Adjuvant_*), 以便对术前病人也适用;
# 术后病人若想把这几个字段纳入训练, 加 --with-treatment 或 WITH_TREATMENT=1。
#
# 环境变量(可选):
#   MODEL=LogReg-L1       # 不跑比拼, 直接用指定算法
#   WITH_TREATMENT=1      # 把辅助治疗字段纳入特征
#
# 例:
#   bash run_all.sh my_patient.yaml                        # 默认: 不含治疗字段
#   MODEL=RandomForest bash run_all.sh my_patient.yaml     # 固定 RF
#   bash run_all.sh my_patient.yaml --with-treatment       # 术后病人模式
#
set -e

cd "$(dirname "$0")"

PATIENT="${1:-patient_template.yaml}"
shift || true

MODEL_ARG=""
TREATMENT_ARG=""                                 # 默认不传 => model.py 默认为 --with-treatment
if [[ -n "${MODEL:-}" ]]; then
  MODEL_ARG="--model $MODEL"
fi
if [[ "${WITH_TREATMENT:-0}" == "1" ]]; then
  TREATMENT_ARG="--with-treatment"
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL_ARG="--model $2"; shift 2;;
    --with-treatment) TREATMENT_ARG="--with-treatment"; shift;;
    --no-treatment) TREATMENT_ARG="--no-treatment"; shift;;
    *) echo "未知参数: $1"; exit 1;;
  esac
done

echo "============================================================"
echo "[1/4] 训练 SNF 分类模型"
echo "  病人文件:      $PATIENT"
echo "  模型选择:      ${MODEL_ARG:-(不指定 => 跑 16 模型大比拼, 自动选最佳)}"
echo "  辅助治疗字段:  ${TREATMENT_ARG:-(默认不含; --with-treatment 可加入)}"
echo "============================================================"
python3 src/model.py $MODEL_ARG $TREATMENT_ARG

echo
echo "============================================================"
echo "[2/4] 预测病人 SNF 亚型"
echo "============================================================"
python3 src/predict_patient.py --patient "$PATIENT"

echo
echo "============================================================"
echo "[3/4] 在原队列里找相似病人(按特征重要性加权欧氏距离)"
echo "============================================================"
python3 src/find_similar.py --patient "$PATIENT" --k 20

echo
echo "============================================================"
echo "[4/5] 画 SNF 与相似病人的 KM 生存曲线"
echo "============================================================"
python3 src/survival_compare.py --patient "$PATIENT" --k 20

echo
echo "============================================================"
echo "[5/5] CoxPH 个人生存预测 (OS / RFS / DMFS, 默认含辅助治疗)"
echo "============================================================"
python3 src/survival_predict.py --patient "$PATIENT"

echo
echo "全部完成, 结果在 outputs/ 目录:"
ls -1 outputs/
