#!/bin/bash
# Complete Installation and Training Script
# Run this to set up PyTorch and train your first model

echo "======================================================================="
echo "Antibody Generator - Installation and Training"
echo "======================================================================="
echo ""

# Navigate to project directory
cd /mnt/c/Users/401-24/Desktop/Ab_generative_model

# Step 1: Install pip if needed
echo "Step 1: Checking pip installation..."
if ! command -v pip3 &> /dev/null; then
    echo "  pip3 not found. Installing pip..."
    sudo apt-get update
    sudo apt-get install -y python3-pip
else
    echo "  ✅ pip3 already installed"
fi

# Step 2: Install PyTorch
echo ""
echo "Step 2: Installing PyTorch (this may take a few minutes)..."
pip3 install torch --index-url https://download.pytorch.org/whl/cpu --quiet
echo "  ✅ PyTorch installed"

# Step 3: Verify installation
echo ""
echo "Step 3: Verifying PyTorch installation..."
python3 -c "import torch; print(f'  ✅ PyTorch {torch.__version__} ready!')"

# Step 4: Test tokenizer
echo ""
echo "Step 4: Testing tokenizer..."
python3 generators/tokenizer.py | tail -5

# Step 5: Test LSTM model
echo ""
echo "Step 5: Testing LSTM model..."
python3 generators/lstm_seq2seq.py | tail -10

# Step 6: Train tiny model (10 minutes)
echo ""
echo "Step 6: Training tiny model (Stage 1 - ~10 minutes)..."
echo "  This will train on 1,000 samples to verify everything works."
echo "  Press Ctrl+C if you want to skip this and run it manually later."
echo ""
read -p "  Start training? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python3 scripts/train_generative.py --stage tiny
else
    echo "  Skipped. Run manually with:"
    echo "  python3 scripts/train_generative.py --stage tiny"
fi

# Done
echo ""
echo "======================================================================="
echo "✅ Setup Complete!"
echo "======================================================================="
echo ""
echo "Next steps:"
echo "  1. Check results in: models/generative/tiny/"
echo "  2. If tiny training succeeded, run:"
echo "     python3 scripts/train_generative.py --stage small"
echo "  3. Then for production:"
echo "     python3 scripts/train_generative.py --stage full"
echo ""
echo "======================================================================="
