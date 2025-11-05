# Training olmocr - Quick Start

## What You Need

**Hardware:**
- **Minimum**: 1x A100 40GB (~$3.67/hr on Azure NC24ads_A100_v4)
- **Recommended**: 1x A100 80GB or H100

**Your M60s won't work** - need compute capability 8.0+ for FP8 model

---

## Copy-Paste Commands

### Step 1: Setup Training Environment

```bash
# Run the setup script
./quick_train_setup.sh

# This creates:
# ~/olmocr_training/
#   ??? data/raw/      - Put your data here
#   ??? data/train/    - Auto-generated
#   ??? data/eval/     - Auto-generated
#   ??? configs/       - Training config
#   ??? checkpoints/   - Saved during training
#   ??? models/        - Final models
```

---

### Step 2: Prepare Training Data

**Option A: Generate from Your PDFs (Easiest)**

```bash
# 1. Convert your PDFs using cloud inference
python -m olmocr.pipeline ./workspace \
  --server https://api.deepinfra.com/v1/openai \
  --api_key YOUR_API_KEY \
  --model allenai/olmOCR-2-7B-1025 \
  --pdfs /path/to/your/*.pdf

# 2. Review/fix the output (manually edit files in ./workspace/)

# 3. Convert to training format
python -m olmocr.data.prepare_workspace \
  ./workspace \
  ~/olmocr_training/data/raw

# 4. Split into train/eval
cd ~/olmocr_training
python prepare_my_data.py data/raw
```

**Option B: Manual Annotation**

```bash
# 1. Split your multi-page PDFs into single pages
# 2. Create matching .md files for each PDF
# 3. Put in ~/olmocr_training/data/raw/
# 4. Split the data
cd ~/olmocr_training
python prepare_my_data.py data/raw
```

---

### Step 3: Train the Model

```bash
cd ~/olmocr_training
./start_training.sh

# Training time estimates:
# - 100 pages: 30-60 minutes
# - 1000 pages: 3-6 hours
# - 10000 pages: 1-2 days
```

---

### Step 4: Prepare Your Trained Model

```bash
# 1. Find best checkpoint
ls ~/olmocr_training/checkpoints/

# 2. Prepare checkpoint (merges LoRA weights)
python -m olmocr.train.prepare_checkpoint \
  ~/olmocr_training/checkpoints/checkpoint-300 \
  ~/olmocr_training/models/my_model

# 3. Quantize to FP8 (optional, makes it 2x smaller & faster)
python -m olmocr.train.compress_checkpoint \
  --config olmocr/train/quantization_configs/qwen2_5vl_w8a8_fp8.yaml \
  ~/olmocr_training/models/my_model \
  ~/olmocr_training/models/my_model_FP8
```

---

### Step 5: Test Your Custom Model

```bash
# Download test file
curl -o ~/test.pdf https://olmocr.allenai.org/papers/olmocr_3pg_sample.pdf

# Test with your model
python -m olmocr.pipeline ~/test_output \
  --model ~/olmocr_training/models/my_model_FP8 \
  --markdown \
  --pdfs ~/test.pdf

# Check results
cat ~/test_output/markdown/test.md
```

---

### Step 6: Use Your Model in Production

```bash
# Process documents with your custom model
python -m olmocr.pipeline ./my_workspace \
  --model ~/olmocr_training/models/my_model_FP8 \
  --markdown \
  --pdfs /path/to/documents/*.pdf
```

---

## Data Format Requirements

Each training example needs:

1. **Single-page PDF** (e.g., `doc1.pdf`)
2. **Matching markdown** (e.g., `doc1.md`)

### Markdown format:

```markdown
---
primary_language: en
is_rotation_valid: True
rotation_correction: 0
is_table: False
is_diagram: False
---
Your document text content here...
```

---

## Training Config Adjustments

Edit `~/olmocr_training/configs/my_finetuning.yaml`:

### If You Run Out of Memory:

```yaml
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 64  # Increase this
  gradient_checkpointing: true     # Enable this
  collator_max_token_len: 4096     # Reduce from 8192
```

### If Training is Too Slow:

```yaml
training:
  torch_compile: true  # Enable this
  num_train_epochs: 1  # Reduce epochs

model:
  lora_rank: 4  # Reduce from 8
```

### If Model Overfits:

```yaml
training:
  num_train_epochs: 1  # Reduce from 3

model:
  lora_dropout: 0.2  # Increase from 0.1
```

---

## Cost Estimates (Azure NC24ads_A100_v4 @ $3.67/hr)

| Dataset Size | Training Time | Cost |
|-------------|---------------|------|
| 100 pages | 30-60 min | $2-4 |
| 1000 pages | 3-6 hours | $11-22 |
| 10000 pages | 1-2 days | $88-176 |

**Tip**: Start small (100-200 pages) to validate your pipeline before scaling up!

---

## Troubleshooting

### "CUDA out of memory"
```bash
# Reduce batch size in config:
training:
  gradient_accumulation_steps: 64  # Increase
  gradient_checkpointing: true     # Enable
```

### "No training data found"
```bash
# Make sure you split your data:
cd ~/olmocr_training
python prepare_my_data.py data/raw

# Verify:
ls data/train/*.pdf
ls data/eval/*.pdf
```

### "Model is overfitting"
```bash
# Reduce epochs in config:
training:
  num_train_epochs: 1
```

### Training is too slow
```bash
# Enable torch compile in config:
training:
  torch_compile: true
```

---

## Complete Example Workflow

```bash
# 1. Setup (one-time)
./quick_train_setup.sh

# 2. Generate training data from your PDFs
python -m olmocr.pipeline ./workspace \
  --server https://api.deepinfra.com/v1/openai \
  --api_key YOUR_KEY \
  --model allenai/olmOCR-2-7B-1025 \
  --pdfs /path/to/100_pdfs/*.pdf

# 3. Prepare for training
python -m olmocr.data.prepare_workspace ./workspace ~/olmocr_training/data/raw
cd ~/olmocr_training
python prepare_my_data.py data/raw

# 4. Train (takes ~1 hour for 100 pages)
./start_training.sh

# 5. Prepare model
python -m olmocr.train.prepare_checkpoint \
  checkpoints/checkpoint-300 models/my_model

python -m olmocr.train.compress_checkpoint \
  --config olmocr/train/quantization_configs/qwen2_5vl_w8a8_fp8.yaml \
  models/my_model models/my_model_FP8

# 6. Test
python -m olmocr.pipeline ./test \
  --model models/my_model_FP8 \
  --markdown \
  --pdfs test.pdf

# 7. Deploy
python -m olmocr.pipeline ./production \
  --model models/my_model_FP8 \
  --markdown \
  --pdfs production_docs/*.pdf
```

---

## Best Practices

1. ? **Start small**: 100-200 pages to validate
2. ? **Clean data**: Quality > quantity
3. ? **Monitor eval loss**: Stop if it increases
4. ? **Save checkpoints**: Keep multiple versions
5. ? **Test frequently**: Validate on real docs
6. ? **Version control**: Track your configs

---

## Next Steps

- **See `TRAINING_GUIDE.md`** for detailed explanation
- **Check `~/olmocr_training/README.txt`** after running setup
- **Join Discord** for help: https://discord.gg/sZq3jTNVNG

Good luck! ??
