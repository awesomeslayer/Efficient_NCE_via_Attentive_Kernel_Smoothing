#!/bin/bash

# Check if at least one seed is provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <dataset_name> <seed1> [seed2] [seed3] ..."
    echo "Example: $0 ETTm1 42 123 456"
    exit 1
fi

# First argument is dataset name, rest are seeds
dataset_name="$1"
shift
seeds=("$@")

# Iterate over each seed
for seed in "${seeds[@]}"; do
    # Run for ConvCDE with gaussian interpolation
    python3 run_forecasting.py \
    --csv_path "data/${dataset_name}.csv" \
    --model_type mvc \
    --context_len 24 \
    --horizon_len 24 \
    --hidden_channels 32 \
    --interp_type gp \
    --batch_size 128 \
    --bandwidth 16 \
    --n_heads 4 \
    --lr 1e-4 \
    --tol 1e-3 \
    --num_epochs 10 \
    --irregularity_fraction 0.7 \
    --seed "$seed"

    # Run for GRU
    python3 run_forecasting.py \
        --csv_path "data/${dataset_name}.csv" \
        --model_type gru \
        --context_len 24 \
        --horizon_len 24 \
        --hidden_channels 128 \
        --batch_size 128 \
        --lr 1e-4 \
        --num_epochs 10 \
        --irregularity_fraction 0.7 \
        --seed "$seed"

    # Run for NeuralCDE with cubic interpolation
    python3 run_forecasting.py \
        --csv_path "data/${dataset_name}.csv" \
        --model_type cde \
        --context_len 24 \
        --horizon_len 24 \
        --hidden_channels 64 \
        --interp_type cubic \
        --batch_size 128 \
        --lr 1e-4 \
        --tol 1e-3 \
        --num_epochs 10 \
        --irregularity_fraction 0.7 \
        --seed "$seed"

done
