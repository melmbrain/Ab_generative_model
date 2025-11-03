#!/bin/bash
# ONE COMMAND TO RULE THEM ALL!
# This installs PyTorch and starts training

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║          Installing PyTorch and Training Your First Model           ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# Navigate to project
cd /mnt/c/Users/401-24/Desktop/Ab_generative_model

# Step 1: Install pip
echo "Step 1/5: Installing pip..."
sudo apt-get update -qq
sudo apt-get install -y python3-pip -qq
echo "✅ pip installed"
echo ""

# Step 2: Install PyTorch
echo "Step 2/5: Installing PyTorch (this takes a few minutes)..."
pip3 install torch --index-url https://download.pytorch.org/whl/cpu --quiet --user
echo "✅ PyTorch installed"
echo ""

# Step 3: Verify installation
echo "Step 3/5: Verifying PyTorch..."
python3 -c "import torch; print(f'✅ PyTorch {torch.__version__} ready!')"
echo ""

# Step 4: Test model
echo "Step 4/5: Testing LSTM model..."
python3 generators/lstm_seq2seq.py 2>&1 | tail -5
echo ""

# Step 5: Train tiny model
echo "Step 5/5: Training tiny model (10 minutes)..."
echo "Starting training on 1,000 samples..."
echo ""
python3 scripts/train_generative.py --stage tiny

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                          🎉 ALL DONE! 🎉                              ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Your first model is trained!"
echo "Check results: models/generative/tiny/best_model.pth"
echo ""
echo "Next steps:"
echo "  - Train small model: python3 scripts/train_generative.py --stage small"
echo "  - Train full model:  python3 scripts/train_generative.py --stage full"
echo ""
