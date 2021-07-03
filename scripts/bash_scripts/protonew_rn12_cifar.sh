for ARGUMENT in "$@"

do

    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)

    case "$KEY" in
            GPU)          GPU=${VALUE} ;;
            SEED)         SEED=${VALUE} ;;
            *)
    esac

    echo "$KEY = $VALUE"

done

for s in ${SEED}; do
        OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=${GPU} python start_training.py --arch resnet12 --batch-size 512 --expm-id resnet12-proto-32w-5s-11q-seed${s}-CIFARFS-240epochs --seed ${s} --dataset CIFARFS --proto-train --proto-train-way 32 --proto-train-shot 5 --proto-train-query 11 --proto-train-iter 75 --epochs 240;
done
