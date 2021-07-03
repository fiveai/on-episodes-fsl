#!/bin/bash

for ARGUMENT in "$@"

do

    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)

    case "$KEY" in
            GPU)          GPU=${VALUE} ;;
            SEED)         SEED=${VALUE} ;;
            DATASET)      DATASET=${VALUE} ;;
            N_WAY)        N_WAY=${VALUE} ;;
            N_SHOT)       N_SHOT=${VALUE} ;;
            N_QUERY)      N_QUERY=${VALUE} ;;
            BATCH)        BATCH=${VALUE} ;;
            EPISODES)     EPISODES=${VALUE} ;;
            *)
    esac

    echo "$KEY = $VALUE"

done

echo $GPU
echo $SEED
echo $DATASET
echo $BATCH
echo $N_WAY
echo $N_SHOT
echo $N_QUERY
echo $EPISODES


for s in ${SEED}; do
        OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=${GPU} python start_training.py --arch resnet12 --batch-size ${BATCH} --seed ${s} --expm-id resnet12-proto-${N_SHOT}shot-${DATASET}-${N_WAY}w-${N_QUERY}q-batch${BATCH}-seed${s} --dataset ${DATASET} --proto-train-iter ${EPISODES} --proto-train-way ${N_WAY} --proto-train-shot ${N_SHOT} --proto-train-query ${N_QUERY} --proto-train;
done

