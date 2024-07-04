#!/bin/bash
. /lustre/fs2/portfolios/nvr/users/jiaruix/miniconda/etc/profile.d/conda.sh

conda activate timm
which python

export WANDB_API_KEY=18a953cf069a567c46b1e613f940e6eb8f878c3d

torchrun --nproc_per_node $SUBMIT_GPUS --master_addr $MASTER_ADDR --master_port $MASTER_PORT --nnodes $NUM_NODES --node_rank $NODE_RANK train.py $@