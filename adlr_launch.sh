#!/bin/bash
# submit_job --gpu 8 --nodes 2 --cpu 248 --partition grizzly --command 'bash slurm_torchrun.sh -c configs/baselines/vit_small_patch16_rope_gap_224.sbb_in1k.yaml  --log-wandb --auto-resume --tag ep90' --logroot /lustre/fs2/portfolios/nvr/users/jiaruix/code/pytorch-image-models/logs --name timm-8gpus-2nodes --duration 4 --duration_min 2.5

# Get the directory of the current script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SCRIPT_DIR="$(realpath "$SCRIPT_DIR")"

submit_job --gpu 8 --nodes 2 --cpu 248 --partition grizzly,polar,polar2,polar3,polar4 --command "bash slurm_torchrun.sh $*" --logroot ${SCRIPT_DIR}/output/logs --name timm-8gpus-2nodes --duration 4 --autoresume_method requeue --autoresume_before_timelimit 5