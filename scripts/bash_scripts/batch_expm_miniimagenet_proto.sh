# NCA
./bash_scripts/generic_nca.sh GPU="6,7" SEED="0 1 2" DATASET=miniimagenet BATCH=128
./bash_scripts/generic_nca.sh GPU="6,7" SEED="0 1 2" DATASET=miniimagenet BATCH=256
./bash_scripts/generic_nca.sh GPU="6,7" SEED="0 1 2" DATASET=miniimagenet BATCH=512


# 5-shot experiments
# Protonets - batch size 128
./bash_scripts/generic_proto.sh GPU="6,7" SEED="0 1 2" DATASET=miniimagenet BATCH=128 EPISODES=300 N_SHOT=5 N_WAY=16 N_QUERY=3
./bash_scripts/generic_proto.sh GPU="6,7" SEED="0 1 2" DATASET=miniimagenet BATCH=128 EPISODES=300 N_SHOT=5 N_WAY=8 N_QUERY=11
./bash_scripts/generic_proto.sh GPU="6,7" SEED="0 1 2" DATASET=miniimagenet BATCH=128 EPISODES=300 N_SHOT=5 N_WAY=4 N_QUERY=27

# Protonets - batch size 256
./bash_scripts/generic_proto.sh GPU="6,7" SEED="0 1 2" DATASET=miniimagenet BATCH=256 EPISODES=150 N_SHOT=5 N_WAY=32 N_QUERY=3
./bash_scripts/generic_proto.sh GPU="6,7" SEED="0 1 2" DATASET=miniimagenet BATCH=256 EPISODES=150 N_SHOT=5 N_WAY=16 N_QUERY=11
./bash_scripts/generic_proto.sh GPU="6,7" SEED="0 1 2" DATASET=miniimagenet BATCH=256 EPISODES=150 N_SHOT=5 N_WAY=8 N_QUERY=27

 # Protonets - batch size 512
./bash_scripts/generic_proto.sh GPU="6,7" SEED="0 1 2" DATASET=miniimagenet BATCH=512 EPISODES=75 N_SHOT=5 N_WAY=64 N_QUERY=3
./bash_scripts/generic_proto.sh GPU="6,7" SEED="0 1 2" DATASET=miniimagenet BATCH=512 EPISODES=75 N_SHOT=5 N_WAY=32 N_QUERY=11
./bash_scripts/generic_proto.sh GPU="6,7" SEED="0 1 2" DATASET=miniimagenet BATCH=512 EPISODES=75 N_SHOT=5 N_WAY=16 N_QUERY=27


# 1-shot experiments
# from the paper
# bs 128
./bash_scripts/generic_proto.sh GPU="6,7" SEED="0 1 2" DATASET=miniimagenet BATCH=128 EPISODES=300 N_SHOT=1 N_WAY=16 N_QUERY=7
# bs 256
./bash_scripts/generic_proto.sh GPU="6,7" SEED="0 1 2" DATASET=miniimagenet BATCH=256 EPISODES=150 N_SHOT=1 N_WAY=32 N_QUERY=7
# bs 512
./bash_scripts/generic_proto.sh GPU="6,7" SEED="0 1 2" DATASET=miniimagenet BATCH=512 EPISODES=75 N_SHOT=1 N_WAY=64 N_QUERY=7

#rest
# Protonets - batch size 128
./bash_scripts/generic_proto.sh GPU="6,7" SEED="0 1 2" DATASET=miniimagenet BATCH=128 EPISODES=300 N_SHOT=1 N_WAY=8 N_QUERY=15
./bash_scripts/generic_proto.sh GPU="6,7" SEED="0 1 2" DATASET=miniimagenet BATCH=128 EPISODES=300 N_SHOT=1 N_WAY=4 N_QUERY=31

# Protonets - batch size 256
./bash_scripts/generic_proto.sh GPU="6,7" SEED="0 1 2" DATASET=miniimagenet BATCH=256 EPISODES=150 N_SHOT=1 N_WAY=16 N_QUERY=15
./bash_scripts/generic_proto.sh GPU="6,7" SEED="0 1 2" DATASET=miniimagenet BATCH=256 EPISODES=150 N_SHOT=1 N_WAY=8 N_QUERY=31

 # Protonets - batch size 512
./bash_scripts/generic_proto.sh GPU="6,7" SEED="0 1 2" DATASET=miniimagenet BATCH=512 EPISODES=75 N_SHOT=1 N_WAY=32 N_QUERY=15
./bash_scripts/generic_proto.sh GPU="6,7" SEED="0 1 2" DATASET=miniimagenet BATCH=512 EPISODES=75 N_SHOT=1 N_WAY=16 N_QUERY=31
