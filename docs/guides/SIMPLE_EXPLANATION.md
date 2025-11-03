# Simple Explanation - How Everything Works

## 🎯 The Big Picture

**Goal**: Train an AI model to generate antibody sequences that bind to specific antigens.

**Think of it like**: Teaching a robot to write recipes. You show it thousands of recipes (training data), and then it learns to write new recipes when you tell it what dish you want.

---

## 📦 The Components (Step by Step)

### 1. **Data** - The Training Examples

**Location**: `data/generative/`
- `train.json` - 126,508 examples to learn from
- `val.json` - 15,813 examples to test understanding
- `test.json` - 15,814 examples for final evaluation

**What's inside each example?**
```json
{
  "antigen_sequence": "MKTAYIAKQR..." (the target protein)
  "antibody_heavy": "EVQLVES..." (antibody heavy chain)
  "antibody_light": "DIQMTQ..." (antibody light chain)
  "pKd": 8.5 (how strongly they bind)
}
```

**Analogy**: Like a cookbook with recipes showing:
- Ingredient (antigen)
- Recipe steps (antibody)
- How tasty it is (binding strength)

---

### 2. **Tokenizer** - The Translator

**File**: `generators/tokenizer.py`

**What it does**: Converts protein sequences (letters) into numbers that the AI can understand.

**Example**:
```
Input:  "ACDEFGH"  (amino acid letters)
Output: [1, 5, 6, 7, 8, 9, 10, 2]  (numbers)
```

**Why needed?**: AI models work with numbers, not letters. This converts back and forth.

**Vocabulary**:
- 20 amino acids: A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y
- 5 special tokens: `<PAD>`, `<START>`, `<END>`, `<UNK>`, `<SEP>`
- Total: 25 tokens

**Analogy**: Like translating English to numbers so a calculator can process it.

---

### 3. **Data Loader** - The Batch Organizer

**File**: `generators/data_loader.py`

**What it does**:
- Reads the JSON data files
- Groups examples into batches (e.g., 16 at a time)
- Uses tokenizer to convert sequences to numbers
- Pads sequences to same length

**Example**:
```
Read 100 examples → Group into batches of 16 →
Batch 1: 16 examples (tokenized, padded)
Batch 2: 16 examples (tokenized, padded)
...
```

**Why needed?**: Training on one example at a time is slow. Batches are faster.

**Analogy**: Like organizing groceries into shopping bags (16 items per bag) for easier carrying.

---

### 4. **Model** - The AI Brain

**File**: `generators/transformer_seq2seq.py`

**What it does**: The actual AI that learns to generate antibodies.

**Architecture** (simplified):
```
INPUT: Antigen + Target Binding Strength
  ↓
ENCODER: Reads and understands the antigen
  ↓
CONDITIONING: Adjusts based on desired binding strength
  ↓
DECODER: Generates antibody sequence letter by letter
  ↓
OUTPUT: Antibody sequence
```

**How it generates** (step by step):
1. Start with `<START>` token
2. Predict next amino acid (e.g., "E")
3. Use "E" to predict next (e.g., "V")
4. Keep going until `<END>` token
5. Result: "EVQLVES..." (antibody sequence)

**Analogy**: Like a chef that:
1. Looks at ingredients (antigen)
2. Remembers desired taste (binding strength)
3. Writes recipe step-by-step (generates antibody)

**Model Sizes**:
- **Tiny**: 0.95M parameters - Fast, for testing
- **Small**: 5.6M parameters - Good balance
- **Medium**: 40M parameters - Better quality
- **Large**: 100M parameters - Best quality

---

### 5. **Training Script** - The Teacher

**File**: `train.py`

**What it does**: Teaches the model by showing it examples.

**Training Process** (each epoch):

**Step 1: Training Phase**
```
For each batch of 16 examples:
  1. Show model the antigen + desired binding
  2. Model generates antibody
  3. Compare to real antibody (compute error)
  4. Adjust model to reduce error
```

**Step 2: Validation Phase**
```
For each validation batch:
  1. Show model antigen
  2. Model generates antibody
  3. Measure error (but DON'T adjust model)
```

**Step 3: Evaluation**
```
Generate 100 antibodies
Measure quality:
  - Are they valid sequences? (should be 100%)
  - Are they diverse? (should be >70%)
  - Do they look realistic?
```

**Step 4: Save Checkpoint**
```
If validation error improved:
  - Save model to file
  - Record metrics
```

**Analogy**: Like teaching a student:
1. Show homework problems (training)
2. Give practice test (validation)
3. Check their work quality
4. Save progress in notebook (checkpoint)

---

### 6. **Metrics** - The Report Card

**File**: `generators/metrics.py`

**What it tracks**:

**Loss** (Error):
- How wrong the model's predictions are
- Lower is better
- Goal: Get this down to ~1.5-2.0

**Validity**:
- % of sequences with only valid amino acids
- Should be 100%
- Example: "ACDEFGH" = valid, "ACX123" = invalid

**Diversity**:
- % of unique sequences generated
- Should be >70%
- Low diversity = model is repeating itself

**Length**:
- Average sequence length
- Should be ~120 amino acids (heavy chain)

**Analogy**: Like grading a student's work:
- Loss = how many mistakes
- Validity = used correct grammar
- Diversity = variety in answers
- Length = appropriate answer length

---

### 7. **Monitor** - The Dashboard

**File**: `monitor_training.py`

**What it does**: Shows you training progress in real-time.

**Example output**:
```
Epoch 5:
  Train Loss: 2.3
  Val Loss:   2.5
  Validity:   98%
  Diversity:  75%
  Time:       120 seconds

Best epoch so far: Epoch 3 (Val Loss: 2.4)
```

**Why useful**:
- See if training is working
- Know when to stop
- Compare different experiments

**Analogy**: Like checking a student's grade report to see if they're improving.

---

## 🔄 How They All Connect

### Data Flow Diagram

```
DATA FILES (train.json)
    ↓
TOKENIZER (converts letters → numbers)
    ↓
DATA LOADER (organizes into batches)
    ↓
MODEL (learns from batches)
    ↓
TRAINING SCRIPT (manages the learning process)
    ↓
METRICS (measures quality)
    ↓
MONITOR (displays progress)
```

### During Training (Detailed Flow)

```
1. train.py starts
2. Loads tokenizer
3. Loads data files via data_loader.py
4. Creates model from transformer_seq2seq.py
5. FOR EACH EPOCH:
   a. FOR EACH BATCH:
      - Get batch from data_loader
      - Model predicts antibody
      - Compute loss (error)
      - Update model weights
   b. Validate on validation set
   c. Generate samples and evaluate with metrics.py
   d. Log results
   e. Save checkpoint if improved
6. Training complete!
7. Use monitor_training.py to view results
```

---

## 💡 Real Example Walkthrough

Let's trace one training example through the system:

### Input Data (from train.json)
```json
{
  "antigen_sequence": "MKTAYIA",
  "antibody_heavy": "EVQLVES",
  "pKd": 8.5
}
```

### Step 1: Tokenizer
```
Antigen:  "MKTAYIA" → [1, 15, 13, 21, 5, 24, 12, 5, 2]
Antibody: "EVQLVES" → [1, 8, 22, 18, 14, 22, 8, 20, 2]
pKd: 8.5 (stays as number)
```

### Step 2: Data Loader
```
Groups with 15 other examples
Pads to same length:
Antigen:  [1, 15, 13, 21, 5, 24, 12, 5, 2, 0, 0, 0, ...]
Antibody: [1, 8, 22, 18, 14, 22, 8, 20, 2, 0, 0, 0, ...]
```

### Step 3: Model Forward Pass
```
Input: Antigen numbers + pKd (8.5)
Encoder: Processes antigen
Decoder: Generates antibody one token at a time
Output: [1, 8, 22, 18, ...] (predicted antibody)
```

### Step 4: Loss Computation
```
Predicted: [1, 8, 22, 18, 14, 22, 8, 20, 2]
Target:    [1, 8, 22, 18, 14, 22, 8, 20, 2]
Loss: 0.1 (low = good, predictions match target)
```

### Step 5: Model Update
```
Compute gradients based on loss
Update model weights to reduce loss
```

### Step 6: After Many Examples
```
Model learns patterns:
- Which amino acids appear together
- How antigens relate to antibodies
- How binding strength affects sequence
```

---

## 🎓 Key Concepts Explained

### What is "Training"?

**Simple**: Showing the model many examples so it learns patterns.

**Process**:
1. Show example: "For antigen X, antibody Y works well"
2. Model guesses antibody for X
3. Compare guess to real answer
4. Adjust model to be more accurate
5. Repeat 126,508 times!

**Result**: Model learns to generate good antibodies for new antigens.

---

### What is a "Batch"?

**Simple**: A group of examples processed together.

**Why?**: Faster than one-at-a-time processing.

**Example**:
- Batch size 16 = process 16 examples at once
- 100 examples ÷ 16 per batch = 7 batches

---

### What is an "Epoch"?

**Simple**: One complete pass through all training data.

**Example**:
- 126,508 training examples
- Batch size 16
- 1 epoch = 7,907 batches
- 10 epochs = see all data 10 times

**Why multiple epochs?**: Model learns better with repetition.

---

### What is "Loss"?

**Simple**: How wrong the model's predictions are.

**Scale**:
- 4.0+ = Terrible (random guessing)
- 3.0 = Bad (initial training)
- 2.0 = Okay (learning patterns)
- 1.5 = Good (decent predictions)
- 1.0 = Excellent (very accurate)

**Goal**: Get loss as low as possible.

---

### What is a "Checkpoint"?

**Simple**: A saved snapshot of the model.

**Contents**:
- Model's learned weights
- Optimizer state
- Current epoch
- Validation loss

**Why?**:
- Resume training later
- Use best model (not last epoch)
- Recover from crashes

---

## 🚀 Usage in Plain English

### To Train the Model:

**Command**:
```bash
python3 train.py --config small --epochs 20 --name my_first_model
```

**What happens**:
1. Loads 126,508 training examples
2. Creates a "small" model (5.6M parameters)
3. Trains for 20 epochs (~2-4 hours on CPU)
4. Saves checkpoints to `checkpoints/`
5. Writes logs to `logs/my_first_model.jsonl`

**Output files**:
- `checkpoints/my_first_model_epoch5.pt` (best model saved)
- `logs/my_first_model.jsonl` (training progress)

---

### To Monitor Training:

**Command**:
```bash
python3 monitor_training.py logs/my_first_model.jsonl
```

**Shows**:
```
Epoch 10:
  Train Loss: 2.1
  Val Loss: 2.3
  Validity: 99%
  Diversity: 78%

Best: Epoch 7 (Val Loss: 2.2)
```

**Tells you**:
- Is training working? (loss decreasing?)
- Which epoch was best?
- Quality of generated sequences

---

### To Generate Antibodies:

**After training** (using saved model):
```python
# Load model
model = load_checkpoint('checkpoints/my_first_model_epoch5.pt')

# Generate antibody for specific antigen
antigen = "MKTAYIAKQRQISFVKSHFSRQ..."
target_binding = 8.5  # Strong binding

antibody = model.generate(antigen, target_binding)
# Result: "EVQLVESGGGLVQPGGSLRLSC..."
```

---

## 🎯 Simple Checklist

### Before Training:
- ✅ Data files exist (`data/generative/*.json`)
- ✅ All Python files in place
- ✅ PyTorch installed

### Start Training:
```bash
python3 train.py --config small --epochs 20 --name test1
```

### Monitor Progress:
```bash
# In another terminal
python3 monitor_training.py logs/test1.jsonl
```

### After Training:
- ✅ Check logs (training successful?)
- ✅ Find best checkpoint (lowest val loss)
- ✅ Use model for generation

---

## 🤔 Common Questions

### Q: Which file do I run?
**A**: `train.py` - This starts everything.

### Q: How long does training take?
**A**:
- Test (100 samples): 30 seconds
- Small (10k samples): 1-2 hours
- Full (127k samples): 4-8 hours on GPU

### Q: How do I know if it's working?
**A**: Loss should decrease. Use `monitor_training.py` to check.

### Q: What if I stop training?
**A**: Checkpoints are saved. You can resume or use best checkpoint.

### Q: Which model size should I use?
**A**:
- Testing: `tiny`
- Development: `small`
- Production: `medium` or `large` (needs GPU)

### Q: What's a good loss value?
**A**: Under 2.0 is good, under 1.5 is great.

---

## 📋 File Purpose Summary

| File | What It Does | When You Use It |
|------|-------------|----------------|
| `train.py` | Trains the model | Start training |
| `monitor_training.py` | Shows progress | Check how training is going |
| `tokenizer.py` | Converts sequences | (Used by train.py automatically) |
| `data_loader.py` | Loads data | (Used by train.py automatically) |
| `transformer_seq2seq.py` | The AI model | (Used by train.py automatically) |
| `metrics.py` | Tracks quality | (Used by train.py automatically) |

**You mainly interact with**: `train.py` and `monitor_training.py`

Everything else works behind the scenes!

---

## 🎉 Summary

**The whole system in one sentence**:

> train.py loads data (using data_loader.py), converts it to numbers (using tokenizer.py), feeds it to the AI (transformer_seq2seq.py), tracks quality (using metrics.py), and saves progress - all while you watch using monitor_training.py.

**Your role**:
1. Run `train.py`
2. Wait (hours)
3. Check results with `monitor_training.py`
4. Use trained model to generate antibodies

That's it! 🚀

---

**Questions?** Ask about any specific component and I'll explain in more detail!
