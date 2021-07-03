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
        OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=${GPU} python start_training.py --arch resnet12 --batch-size 512 --expm-id resnet12-nca-proj128-seed${s}-batch512-temp0.9-miniimagenet-soft --seed ${s} --temperature 0.9 --dataset miniimagenet --projection --projection-feat-dim 128 --soft-assignment;
done
