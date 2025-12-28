#!/bin/bash
cfgs=$1
pretrained=$2
expname=$3
gpus=$4
if [ $# -ge 5 ]; then
    # if the 7th arg is provided, then it is used as batch_size
    batch_size=$5
else
    # otherwise, use the default.
    batch_size=4
fi

python3 -m torch.distributed.launch \
    --nproc_per_node=$gpus \
    --nnodes 1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=20596 \
    evaluate.py --batch-size $batch_size --data-root PATH_TO_DATA_ROOT --ftype local --name SlPromptDA \
    --dsname dav2-fix --pretrained $pretrained --cfgs-path $cfgs --exproot PATH_TO_EXP_OUTPUT_DIR \
    --expname $expname --sample-rate 0.2 --split test --num-workers 1 \
    --metrics 'rmse,mae,absrel,delta_125,delta_110,delta_105' --load-pattern \
    --per-mat-metrics
