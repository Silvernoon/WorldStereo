#!/bin/bash
# WorldStereo WAM training script with DeepSpeed ZeRO-1
# Similar to FastWAM's train_zero1.sh

set -e

# Default values
CONFIG_NAME="train"
TASK=""
DATA=""
MODEL=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --task)
            TASK="task=$2"
            shift 2
            ;;
        --data)
            DATA="data=$2"
            shift 2
            ;;
        --model)
            MODEL="model=$2"
            shift 2
            ;;
        *)
            # Pass through other arguments
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

echo "==================================="
echo "WorldStereo WAM Training (ZeRO-1)"
echo "==================================="
echo "Task: ${TASK:-default}"
echo "Data: ${DATA:-default}"
echo "Model: ${MODEL:-default}"
echo "Extra args: ${EXTRA_ARGS}"
echo "==================================="

accelerate launch \
    --config_file scripts/accelerate_configs/accelerate_zero1.yaml \
    scripts/train.py \
    ${TASK} \
    ${DATA} \
    ${MODEL} \
    ${EXTRA_ARGS}
