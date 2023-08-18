#!/bin/bash
TIME=$(date "+%Y-%m-%d-%H-%M-%S")
## 设置环境变量RANK
#export RANK=0
#export LOCAL_RANK=-1
OUTPUT_PATH=./outputs
TRAIN_LIST=/media/spgou/DATA/ZYJ/Glioma_DL/dataset/train_patients.txt
VAL_LIST=/media/spgou/DATA/ZYJ/Glioma_easy/dataset/test_patients.txt
    

#CUDA_VISIBLE_DEVICES=0 \
#torchrun train_val.py \
#    --lr  0.01 --epochs 500  --batch-size 4  -j 4 \
#    --output=$OUTPUT_PATH/$TIME \
#    --train_list=$TRAIN_LIST \
#    --val_list=$VAL_LIST \
#    --num_classes=4 \

#CUDA_VISIBLE_DEVICES=1 \
#torchrun --master_port=25641 \
#    evaluation.py \
#    --batch-size 1  -j 4 \
#    --output=$OUTPUT_PATH/$TIME \
#    --val_list=$VAL_LIST \
#    --tune_from='/media/spgou/DATA/ZYJ/Glioma_easy/outputs/precision_0.5805_num_80/epoch_80.pth' \
#    --num_classes=4 \




#train
# CUDA_VISIBLE_DEVICES=0 \
# python3 -u -m torch.distributed.launch --nproc_per_node 4  ./tools/train_val.py \
#     --lr  0.01 --epochs 500  --batch-size 2  -j 4 \
#     --output=$OUTPUT_PATH/$TIME \
#     --train_list=$TRAIN_LIST \
#     --val_list=$VAL_LIST \
#     --num_classes=4 \
#     --local_rank=$LOCAL_RANK


# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# python3 -u -m torch.distributed.launch --nproc_per_node 1  ./tools/evaluation.py \
#     --model_name=resnet18 \
#     --batch-size 64  -j 4 \
#     --output=$OUTPUT_PATH/$TIME \
#     --val_list=$VAL_LIST \
#     --tune_from='/home/lxztju/pytorch_classification/ouputs/xxx/epoch_4.pth' \
#     --num_classes=2
    

# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# python3 -u -m torch.distributed.launch --nproc_per_node 1  ./tools/predict.py \
#     --model_name=resnet18 \
#     --batch-size 64  -j 4 \
#     --output=$OUTPUT_PATH/$TIME \
#     --val_list=$VAL_LIST \
#     --tune_from='/home/lxztju/pytorch_classification/ouputs/xxx/epoch_4.pth' \
#     --num_classes=2
