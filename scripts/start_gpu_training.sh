#!/bin/bash

# GPU Training Script - Optimized for RTX 2060
# Training on all 127k samples with GPU acceleration

echo "=========================================="
echo "Starting GPU Training (RTX 2060)"
echo "=========================================="
echo "Dataset: 126,508 training samples"
echo "Config: medium (40M parameters)"
echo "Device: CUDA (GPU)"
echo "Batch Size: 32"
echo "Estimated time: 1-2 hours"
echo "=========================================="
echo ""

# Check GPU before starting
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Start training
python3 train.py \
    --config medium \
    --epochs 20 \
    --batch-size 32 \
    --device cuda \
    --lr 1e-4 \
    --eval-interval 2 \
    --early-stopping 5 \
    --name production_gpu_v1

echo ""
echo "=========================================="
echo "Training Complete!"
echo "Check results with:"
echo "  python3 monitor_training.py logs/production_gpu_v1.jsonl"
echo "=========================================="
