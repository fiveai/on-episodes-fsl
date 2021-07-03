for ARGUMENT in "$@"

do

    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)

    case "$KEY" in
            GPU)          GPU=${VALUE} ;;
            *)
    esac

    echo "$KEY = $VALUE"

done

# Expm IDs needed to reevaluate for matching networks
EXPM_IDS=("20-12-02_01-30-04resnet12-proto-no-aggregates-5shot-CIFARFS-16w-3q-batch128-seed0"

          #NCA replacement
          "20-12-01_12-42-10resnet12-nca-replacement-CIFARFS-batch128-seed0"

          # fixed batch
          "20-12-01_19-19-05resnet12-nca-fixed-batch-5shot-CIFARFS-16w-3q-batch128-seed0"

          # NCA
          "20-10-23_21-22-06resnet12-nca-CIFARFS-batch128-seed0")

# New unique Expm IDs needed to reevaluate for matching networks
NEW_EXPM_IDS=("resnet12-proto-no-aggregates-5shot-CIFARFS-16w-3q-batch128-seed0-matching"

          #NCA replacement
          "resnet12-nca-replacement-CIFARFS-batch128-seed0-matching"

          # fixed batch
          "resnet12-nca-fixed-batch-5shot-CIFARFS-16w-3q-batch128-seed0-matching"

          # NCA
          "resnet12-nca-CIFARFS-batch128-seed0-matching")


# rerun all the models and save results with new expm id
for ((i=0;i<14;++i)); do
    echo ${NEW_EXPM_IDS[i]}
    echo ${EXPM_IDS[i]}
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=${GPU} python start_training.py --arch resnet12 --batch-size 128 --expm-id ${NEW_EXPM_IDS[i]} --evaluate-model ${EXPM_IDS[i]} --dataset CIFARFS --soft-assignment --val-iter 10000;
done
