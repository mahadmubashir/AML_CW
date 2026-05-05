#!/bin/bash
# launches a training run with focal loss instead of cross-entropy
# expects a python env with project requirements active

set -e

export EXPERIMENT_NAME=mobilenetv3_large_run4_focal_loss
export USE_FOCAL_LOSS=1

python -u train_script.py
