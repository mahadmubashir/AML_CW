#!/bin/bash
# launches a training run with a lower learning rate
# expects a python env with project requirements active

set -e

export EXPERIMENT_NAME=mobilenetv3_large_run2_lr_low
export LR_OVERRIDE=1e-4

python -u train_script.py
