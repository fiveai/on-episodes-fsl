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
        OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=${GPU} python start_training.py --arch resnet12 --batch-size 512 --expm-id resnet12-proto-epochs150-oldsetup-20w-5s-15q-439iter-seed${s}-milestones05075-tieredImagenet --seed ${s} --epochs 150 --dataset tieredimagenet --lr-milestones "0.5, 0.75" --proto-train --proto-train-way 20 --proto-train-shot 5 --proto-train-query 15 --proto-train-iter 439;
done
