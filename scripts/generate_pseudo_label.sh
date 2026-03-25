# #!/bin/bash

# DATASET=$1
# # 定义项目根目录，方便后续调用
# PROJ_ROOT="/data/ZhaoX/OVM3D-Det-1"

# # # Step 1: Predict depth using UniDepth
# CUDA_VISIBLE_DEVICES=0 python ${PROJ_ROOT}/third_party/UniDepth/run_unidepth.py --dataset $DATASET

# # Step 2: Segment novel objects using Grounded-SAM
# CUDA_VISIBLE_DEVICES=0 python ${PROJ_ROOT}/third_party/Grounded-Segment-Anything/grounded_sam_detect.py --dataset $DATASET
# CUDA_VISIBLE_DEVICES=0 python ${PROJ_ROOT}/third_party/Grounded-Segment-Anything/grounded_sam_detect_ground.py --dataset $DATASET

# # Step 3: Generate pseudo 3D bounding boxes
# # 注意：这里把 config-file 改成了绝对路径
# python ${PROJ_ROOT}/tools/generate_pseudo_bbox.py \
#   --config-file ${PROJ_ROOT}/configs/Base_Omni3D_${DATASET}.yaml \
#   OUTPUT_DIR ${PROJ_ROOT}/output/generate_pseudo_label/$DATASET

# # Step 4: Convert to COCO dataset format
# python ${PROJ_ROOT}/tools/transform_to_coco.py --dataset_name $DATASET


#!/bin/bash
DATASET=$1
PROJ_ROOT="/data/ZhaoX/OVM3D-Det-1"

# ⚠️ 注释掉 Step 1: 我们已经在 process_indoor.py 内部现场生成了更完美的深度！
# CUDA_VISIBLE_DEVICES=0 python ${PROJ_ROOT}/third_party/UniDepth/run_unidepth.py --dataset $DATASET

# Step 2: Grounded-SAM (保持正常运行)
CUDA_VISIBLE_DEVICES=0 python ${PROJ_ROOT}/third_party/Grounded-Segment-Anything/grounded_sam_detect.py --dataset $DATASET
CUDA_VISIBLE_DEVICES=0 python ${PROJ_ROOT}/third_party/Grounded-Segment-Anything/grounded_sam_detect_ground.py --dataset $DATASET

# Step 3: 这时候会调用我们刚替换的 process_indoor.py，彻底起飞！
python ${PROJ_ROOT}/tools/generate_pseudo_bbox.py \
  --config-file ${PROJ_ROOT}/configs/Base_Omni3D_${DATASET}.yaml \
  OUTPUT_DIR ${PROJ_ROOT}/output/generate_pseudo_label/$DATASET

# Step 4: 转换格式，正常使用
python ${PROJ_ROOT}/tools/transform_to_coco.py --dataset_name $DATASET