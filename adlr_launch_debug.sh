#!/bin/bash

set -x # Enable debugging

# command="bash slurm_torchrun.sh $@"

# echo submit_job --gpu 8 --nodes 1 --cpu 248 --partition grizzly --command "${command}" --logroot /lustre/fs2/portfolios/nvr/users/jiaruix/code/pytorch-image-models/logs --name timm-8gpus-1nodes --duration 4 --autoresume_method requeue --autoresume_before_timelimit 5 --autoresume_timer 10

echo submit_job --gpu 8 --nodes 1 --cpu 248 --partition grizzly --command "bash slurm_torchrun.sh $*" --logroot /lustre/fs2/portfolios/nvr/users/jiaruix/code/pytorch-image-models/logs --name timm-8gpus-1nodes --duration 4 --autoresume_method requeue --autoresume_before_timelimit 5 --autoresume_timer 10