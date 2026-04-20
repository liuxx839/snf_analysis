#!/usr/bin/env bash
# 一键: 模型大比拼(自动选最佳) -> 预测病人 -> 找相似病人 -> 画 KM 曲线
#
# 用法:
#   bash run_all.sh [patient_yaml] [--model MODEL_NAME] [--with-treatment]
#
# 环境变量(可选):
#   MODEL=LogReg-L1       # 不跑比拼直接用指定算法
#   WITH_TREATMENT=1      # 把辅助治疗三字段也加入特征
#
# 例:
#   bash run_all.sh my_patient.yaml                      # 自动选最佳
#   MODEL=RandomForest bash run_all.sh my_patient.yaml   # 固定 RF
#
set -e

cd "$(dirname "$0")"

PATIENT="${1:-patient_template.yaml}"
shift || true

MODEL_ARG=""
WITH_TREATMENT_ARG=""
if [[ -n "${MODEL:-}" ]]; then
  MODEL_ARG="--model $MODEL"
fi
if [[ "${WITH_TREATMENT:-0}" == "1" ]]; then
  WITH_TREATMENT_ARG="--with-treatment"
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL_ARG="--model $2"; shift 2;;
    --with-treatment) WITH_TREATMENT_ARG="--with-treatment"; shift;;
    *) echo "未知参数: $1"; exit 1;;
  esac
done

echo "============================================================"
echo "[1/4] 训练 SNF 分类模型"
echo "  病人文件:      $PATIENT"
echo "  模型选择:      ${MODEL_ARG:-(不指定 => 跑 16 模型大比拼, 自动选最佳)}"
echo "  辅助治疗字段:  ${WITH_TREATMENT_ARG:-(默认不使用)}"
echo "============================================================"
python3 src/model.py $MODEL_ARG $WITH_TREATMENT_ARG

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
echo "[4/4] 画 SNF 与相似病人的 KM 生存曲线"
echo "============================================================"
python3 src/survival_compare.py --patient "$PATIENT" --k 20

echo
echo "全部完成, 结果在 outputs/ 目录:"
ls -1 outputs/
