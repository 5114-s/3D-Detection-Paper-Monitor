#!/bin/bash
#
# 蒸馏训练启动脚本 - SUNRGBD 数据集
# 完整对齐原版 OVM3D-Det 的训练逻辑
#
# 使用方法:
#   bash scripts/train_distill_sunrgbd.sh
#
# 可选参数:
#   GPU_ID   - GPU ID (默认 1)
#   DATASET  - 数据集名 (默认 SUNRGBD)
#   OUTPUT   - 输出目录
#

# 命令行参数：$1=GPU_ID, $2=DATASET, $3=OUTPUT_DIR
GPU_ID=${1:-1}
DATASET=${2:-SUNRGBD}
OUTPUT_DIR=${3:-output/distill_sunrgbd}

# 确保使用正确的 GPU
export CUDA_VISIBLE_DEVICES=$GPU_ID

# 显存优化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 梯度裁剪
MAX_GRAD_NORM=40.0

# 项目根目录
cd "$(dirname "$0")/.." || exit 1

echo "============================================"
echo " 蒸馏训练配置"
echo "============================================"
echo " GPU ID: $GPU_ID"
echo " 数据集: $DATASET"
echo " 输出目录: $OUTPUT_DIR"
echo "============================================"

python tools/train_distill_sunrgbd.py \
  --config-file configs/Base_Omni3D_SUNRGBD.yaml \
  --num-gpus 1 \
  OUTPUT_DIR "$OUTPUT_DIR" \
  DATASETS.TRAIN "(${DATASET}_train,${DATASET}_val,)" \
  DATASETS.FOLDER_NAME Omni3D \
  SOLVER.IMS_PER_BATCH 2 \
  SOLVER.MAX_ITER 15000 \
  SOLVER.BASE_LR 0.001 \
  MODEL.STABILIZE 0.5
