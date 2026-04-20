#!/usr/bin/env bash
# 一键: 训练模型 -> 预测病人 -> 找相似病人 -> 画 KM 曲线
# 用法: bash run_all.sh [patient_yaml]
set -e
PATIENT="${1:-patient_template.yaml}"

cd "$(dirname "$0")"

echo "[1/4] 训练 SNF 分类模型 (5 折 CV)..."
python3 src/model.py

echo
echo "[2/4] 预测病人 SNF 亚型..."
python3 src/predict_patient.py --patient "$PATIENT"

echo
echo "[3/4] 在原队列里找相似病人..."
python3 src/find_similar.py --patient "$PATIENT" --k 20

echo
echo "[4/4] 画 SNF 与相似病人的 KM 生存曲线..."
python3 src/survival_compare.py --patient "$PATIENT" --k 20

echo
echo "全部完成, 结果在 outputs/ 目录:"
ls -1 outputs/
