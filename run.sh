#!/bin/bash

CMD=$1
GPU_ID=$2
CONFIG_PATH=${3:-"./main/config.yaml"}

if [ -n "$GPU_ID" ]; then
  export CUDA_VISIBLE_DEVICES=$GPU_ID
  echo ">>> Using GPU: $GPU_ID (CUDA_VISIBLE_DEVICES=$GPU_ID)"
else
  echo ">>> Using default GPU settings"
fi

EXP_DIRS=$(ls -d experiment_* 2>/dev/null)

case "$CMD" in
  main)
    echo ">>> Running training with config: $CONFIG_PATH"
    python -m main.main --config "$CONFIG_PATH"
    ;;

  agg|aggregate)
    echo ">>> Aggregating all experiment directories..."
    if [ -z "$EXP_DIRS" ]; then
      echo "No experiment_* directories found!"
      exit 1
    fi

    python -m main.plots.aggregate $EXP_DIRS

    for dir in $EXP_DIRS; do
      if [ -f "$dir/summary_results.csv" ]; then
        echo ">>> Generating final table for $dir"
        python -m main.plots.final_table "$dir/summary_results.csv"
      fi
    done
    ;;

  plot|plots)
    echo ">>> Generating all plots into pictures/ directory..."
    mkdir -p pictures
    
    # 1. Ablation plots
    if [ -d "experiment_main" ]; then
        echo ">>> Plotting Ablation..."
        python -m main.plots.ablation --input experiment_main/summary_results.csv --out_dir pictures/ablation
    fi

    # 2. Heads plots
    if [ -d "experiment_heads" ]; then
        echo ">>> Plotting Heads..."
        python -m main.plots.heads --input experiment_heads/summary_results.csv --out_dir pictures/heads
    fi

    # 3. Noise Robustness plots
    if [ -d "experiment_noise_results" ]; then
        echo ">>> Plotting Noise Robustness..."
        python -m main.plots.noise --plots_root experiment_noise_results --output pictures/noise
    fi

    # 4. Time/Speed plots
    if [ -d "experiment_results_time_stats" ]; then
        echo ">>> Plotting Time Stats..."
        python -m main.plots.time --plots_root experiment_results_time_stats --output pictures/time
    fi

    # 5. Attention plots
    if [ -d "experiment_best" ]; then
        echo ">>> Plotting Attention..."
        python -m main.plots.attention --plots_root experiment_best --output pictures/attention
    fi

    echo ">>> All plots generated in the 'pictures/' folder."
    ;;

  *)
    echo "Usage: $0 {main|agg|plot} [GPU_ID] [CONFIG_PATH]"
    echo "Example:"
    echo "  $0 main 0 ./main/config.yaml"
    echo "  $0 agg"
    echo "  $0 plot"
    exit 1
    ;;
esac