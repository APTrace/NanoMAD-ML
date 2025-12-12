#!/bin/bash
# Example inference commands for NanoMAD-ML
# Adjust paths for your system

# 3D inference (default mode)
python src/inference.py \
    -c models/checkpoint_v5.pt \
    --intensity /path/to/intensity.npy \
    --energies /path/to/energies.npy \
    --log-transform-intensity \
    -o inference_output/

# 2D inference (single particle)
python src/inference.py \
    --mode 2d \
    -c models/checkpoint_v5.pt \
    --test-file /path/to/particle.npz \
    --log-transform-intensity \
    -o inference_output_2d/
