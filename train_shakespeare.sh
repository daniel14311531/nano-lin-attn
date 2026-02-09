#!/bin/bash
echo "Starting Shakespeare model training..."

configs=(
    # "gpt2"
    "deltanet"
    "omd_deltanet"
    "conceptual_deltanet"
)

for config in "${configs[@]}"; do
    echo "Training with configuration: $config"
    python train.py config/train_shakespeare_${config}.py
done