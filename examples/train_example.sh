#!/bin/bash
# Example training command for NanoMAD-ML
# Adjust paths for your system

python src/train.py \
    --data-dir /path/to/synthetic_data \
    --output-dir training_output \
    --weight-scheme log \
    --log-transform-intensity \
    --lambda-fa 1.0 \
    --epochs 200 \
    --batch-size 32
