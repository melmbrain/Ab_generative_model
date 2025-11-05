#!/usr/bin/env python3
"""
Status Checker - Verify what's ready and what's pending
"""

import os
import sys
import json
from pathlib import Path

def check_status():
    print("="*70)
    print("Antibody Generator - System Status Check")
    print("="*70)
    print()

    # Check data
    print("📁 Data Preparation:")
    data_dir = Path("data/generative")
    if data_dir.exists():
        train_file = data_dir / "train.json"
        val_file = data_dir / "val.json"
        test_file = data_dir / "test.json"

        if train_file.exists():
            with open(train_file) as f:
                train_data = json.load(f)
            print(f"  ✅ Train data: {len(train_data):,} samples")
        else:
            print(f"  ❌ Train data: NOT FOUND")

        if val_file.exists():
            with open(val_file) as f:
                val_data = json.load(f)
            print(f"  ✅ Val data:   {len(val_data):,} samples")
        else:
            print(f"  ❌ Val data: NOT FOUND")

        if test_file.exists():
            with open(test_file) as f:
                test_data = json.load(f)
            print(f"  ✅ Test data:  {len(test_data):,} samples")
        else:
            print(f"  ❌ Test data: NOT FOUND")
    else:
        print(f"  ❌ Data directory not found")

    # Check PyTorch
    print()
    print("🔧 PyTorch Installation:")
    try:
        import torch
        print(f"  ✅ PyTorch {torch.__version__} installed")
        print(f"  ✅ Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    except ImportError:
        print(f"  ❌ PyTorch NOT installed")
        print(f"     Run: pip3 install torch --index-url https://download.pytorch.org/whl/cpu")

    # Check tokenizer
    print()
    print("🔤 Tokenizer:")
    try:
        sys.path.insert(0, str(Path.cwd()))
        from generators.tokenizer import AminoAcidTokenizer
        tokenizer = AminoAcidTokenizer()
        print(f"  ✅ Tokenizer working")
        print(f"  ✅ Vocabulary size: {tokenizer.vocab_size}")
    except Exception as e:
        print(f"  ❌ Tokenizer error: {e}")

    # Check models
    print()
    print("🧠 Models:")
    try:
        from generators.lstm_seq2seq import create_model
        model = create_model('tiny', vocab_size=25)
        print(f"  ✅ LSTM model code working")
        print(f"  ✅ Tiny model: {model.get_model_size():,} parameters")
    except Exception as e:
        print(f"  ❌ Model error: {e}")

    # Check trained models
    print()
    print("📦 Trained Models:")
    models_dir = Path("models/generative")
    if models_dir.exists():
        for stage in ['tiny', 'small', 'full']:
            stage_dir = models_dir / stage
            if stage_dir.exists():
                best_model = stage_dir / "best_model.pth"
                if best_model.exists():
                    size_mb = best_model.stat().st_size / 1024 / 1024
                    print(f"  ✅ {stage.capitalize()} model trained ({size_mb:.1f} MB)")
                else:
                    print(f"  ⏳ {stage.capitalize()} model: Not trained yet")
            else:
                print(f"  ⏳ {stage.capitalize()} model: Not trained yet")
    else:
        print(f"  ⏳ No models trained yet")

    # Check discriminator
    print()
    print("🎯 Discriminator:")
    disc_model = Path("models/agab_phase2_model.pth")
    if disc_model.exists():
        size_mb = disc_model.stat().st_size / 1024 / 1024
        print(f"  ✅ Phase 2 discriminator ready ({size_mb:.1f} MB)")
    else:
        print(f"  ❌ Discriminator model not found")

    # Summary
    print()
    print("="*70)
    print("Summary:")
    print("="*70)

    # Count what's done
    data_ready = data_dir.exists() and all([
        (data_dir / f).exists() for f in ['train.json', 'val.json', 'test.json']
    ])

    try:
        import torch
        pytorch_ready = True
    except:
        pytorch_ready = False

    try:
        from generators.tokenizer import AminoAcidTokenizer
        from generators.lstm_seq2seq import create_model
        code_ready = True
    except:
        code_ready = False

    trained_models = []
    if models_dir.exists():
        for stage in ['tiny', 'small', 'full']:
            if (models_dir / stage / "best_model.pth").exists():
                trained_models.append(stage)

    # Print summary
    if data_ready:
        print("✅ Data prepared (158k samples)")
    else:
        print("❌ Data NOT prepared")

    if code_ready:
        print("✅ Code working (tokenizer, model)")
    else:
        print("❌ Code NOT working")

    if pytorch_ready:
        print("✅ PyTorch installed")
    else:
        print("❌ PyTorch NOT installed - INSTALL THIS NEXT!")

    if trained_models:
        print(f"✅ Trained models: {', '.join(trained_models)}")
    else:
        print("⏳ No models trained yet")

    # Next steps
    print()
    print("Next Steps:")
    if not pytorch_ready:
        print("  1. Install PyTorch:")
        print("     pip3 install torch --index-url https://download.pytorch.org/whl/cpu")
        print()
        print("  2. Test model:")
        print("     python3 generators/lstm_seq2seq.py")
        print()
        print("  3. Train tiny model:")
        print("     python3 scripts/train_generative.py --stage tiny")
    elif 'tiny' not in trained_models:
        print("  1. Train tiny model (10 minutes):")
        print("     python3 scripts/train_generative.py --stage tiny")
    elif 'small' not in trained_models:
        print("  1. Train small model (1-2 hours):")
        print("     python3 scripts/train_generative.py --stage small")
    elif 'full' not in trained_models:
        print("  1. Train full model (10-20 hours):")
        print("     python3 scripts/train_generative.py --stage full")
    else:
        print("  ✅ All models trained! Ready for production use!")
        print("  1. Generate antibodies for your antigen")
        print("  2. See README.md for usage examples")

    print("="*70)


if __name__ == '__main__':
    check_status()
