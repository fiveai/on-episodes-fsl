#!/bin/bash

for ARGUMENT in "$@"

do

    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)

    case "$KEY" in
            GPU)          GPU=${VALUE} ;;
            SEED)         SEED=${VALUE} ;;
            DATASET)      DATASET=${VALUE} ;;
            *)
    esac

    echo "$KEY = $VALUE"

done

echo $GPU
echo $SEED
echo $DATASET

for s in ${SEED}; do
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=${GPU} python start_training.py --arch resnet12 --batch-size 256 --seed ${s} --expm-id iclr-rebuttal-hyperparameters-resnet12-nca-${DATASET}-seed${s} --dataset ${DATASET} --epochs 90 --lr-milestones "0.5, 0.73" --weight-decay 1e-4;

    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=${GPU} python start_training.py --arch resnet12 --batch-size 256 --seed ${s} --expm-id iclr-rebuttal-hyperparameters-resnet12-proto-${DATASET}-seed${s} --dataset ${DATASET} --epochs 90 --lr-milestones "0.5, 0.73" --weight-decay 1e-4 --proto-train --proto-train-way 16 --proto-train-shot 5 --proto-train-query 11 --proto-train-iter 150;

done
