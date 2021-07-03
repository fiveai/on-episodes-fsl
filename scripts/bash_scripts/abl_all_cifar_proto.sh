./bash_scripts/abl_nca_repl_sampling_128.sh GPU="0,1" SEED="0 1 2 4 5" DATASET=CIFARFS
./bash_scripts/abl_nca_fixed_batch_composition_128.sh GPU="0,1" SEED="0 1 2 3 4" DATASET=CIFARFS
./bash_scripts/abl_proto_no_aggregates_128.sh GPU="0,1" SEED="0 1 2 3 4" DATASET=CIFARFS
./bash_scripts/abl_proto_no_SQ_separation_128.sh GPU="0,1" SEED="0 1 2 3 4" DATASET=CIFARFS

./bash_scripts/abl_nca_repl_sampling.sh GPU="0,1" SEED="0 1 2 3 4" DATASET=CIFARFS
./bash_scripts/abl_nca_fixed_batch_composition.sh GPU="0,1" SEED="0 1 2 3 4" DATASET=CIFARFS
./bash_scripts/abl_proto_no_aggregates.sh GPU="0,1" SEED="0 1 2 3 4" DATASET=CIFARFS
./bash_scripts/abl_proto_no_SQ_separation.sh GPU="0,1" SEED="0 1 2 3 4" DATASET=CIFARFS

./bash_scripts/abl_nca_repl_sampling_128.sh GPU="0,1" SEED="0 1 2 3 4" DATASET=CIFARFS
./bash_scripts/abl_nca_fixed_batch_composition_512.sh GPU="0,1" SEED="0 1 2 3 4" DATASET=CIFARFS
./bash_scripts/abl_proto_no_aggregates_512.sh GPU="0,1" SEED="0 1 2 3 4" DATASET=CIFARFS
./bash_scripts/abl_proto_no_SQ_separation_512.sh GPU="0,1" SEED="0 1 2 3 4" DATASET=CIFARFS
