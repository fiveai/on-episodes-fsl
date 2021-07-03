for ARGUMENT in "$@"

do

    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)

    case "$KEY" in
            GPU)          GPU=${VALUE} ;;
            SAMPLE)         SAMPLE=${VALUE} ;;
            SEED)          SEED=${VALUE} ;;
            *)
    esac

    echo "$KEY = $VALUE"

done

for s in ${SEED}; do
for n in ${SAMPLE}; do
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=${GPU} python start_training.py --expm-id nca-nneg${n}-npos${n}-resnet12-batch256-miniimagenet-softassignment-seed${s} --arch resnet12 --seed ${s} --negatives-frac-random ${n} --positives-frac-random ${n} --batch-size 256 --dataset miniimagenet --soft-assignment;
done
done
