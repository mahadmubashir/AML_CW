#!/bin/bash
# launches a training run with image preprocessing enabled
# expects a python env with project requirements active

set -e

export EXPERIMENT_NAME=mobilenetv3_large_run3_preprocessing
export USE_PREPROCESSING=1

python -u train_script.py
