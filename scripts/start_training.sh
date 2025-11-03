#!/bin/bash

# Full Training Script - CPU Optimized
# Training on all 127k samples

echo "=========================================="
echo "Starting Full Production Training"
echo "=========================================="
echo "Dataset: 126,508 training samples"
echo "Config: small (5.6M parameters)"
echo "Device: CPU"
echo "Estimated time: 6-12 hours"
echo "=========================================="
echo ""

# Start training
python3 train.py \
    --config small \
    --epochs 20 \
    --batch-size 16 \
    --device cpu \
    --lr 1e-4 \
    --eval-interval 2 \
    --early-stopping 5 \
    --name production_full_v1

echo ""
echo "=========================================="
echo "Training Complete!"
echo "Check results with:"
echo "  python3 monitor_training.py logs/production_full_v1.jsonl"
echo "=========================================="
