# Complete Guide: Training olmocr on Your Data

## Overview

There are **2 ways** to train olmocr on your data:

1. **LoRA Fine-tuning** - Fast, works on smaller GPUs (A100 40GB+), great for adapting to your specific documents
2. **Full Training** - Expensive, requires 8xH100, for completely custom models

Most users want **Option 1: LoRA Fine-tuning** ?

---

## Prerequisites

### Hardware Requirements

**For LoRA Fine-tuning (Recommended):**
- **Minimum**: 1x A100 40GB
- **Recommended**: 1x A100 80GB or 1x H100
- **Azure VMs**: NC24ads_A100_v4 (~$3.67/hr)

**For Full Training (Advanced):**
- **Required**: 8x H100 80GB
- **Cost**: ~$300 for 24-48 hours
- **Alternative**: Use cloud training services

### Software Requirements

```bash
# Install training dependencies
pip install olmocr[train]
pip install transformers==4.52.4
pip install flash-attn>=2.8.0.post2 --no-build-isolation
```

---

## Part 1: Prepare Your Training Data

### Data Format

olmocr needs **single-page PDFs** paired with markdown annotations:

```
my_training_data/
??? doc1_page1.pdf
??? doc1_page1.md
??? doc2_page1.pdf
??? doc2_page1.md
??? doc3_page1.pdf
??? doc3_page1.md
??? ...
```

### Markdown Format

Each `.md` file needs YAML front matter + text:

```markdown
---
primary_language: en
is_rotation_valid: True
rotation_correction: 0
is_table: False
is_diagram: False
---
Your document text goes here...
Tables, equations, everything you want the model to learn.
```

---

## Method 1: Use Existing olmocr Output

**Easiest way to get training data!**

```bash
# Step 1: Convert your PDFs with olmocr
python -m olmocr.pipeline ./my_workspace --pdfs /path/to/your/*.pdf

# Step 2: Review and fix the output
# Edit files in ./my_workspace/ to correct any errors

# Step 3: Convert to training format
python -m olmocr.data.prepare_workspace ./my_workspace ./my_training_data
```

Now `./my_training_data` has properly formatted training data!

---

## Method 2: Manual Annotation

If you have ground truth data:

```bash
# 1. Split multi-page PDFs into single pages
# (Use a tool like pdftk or PyPDF2)

# 2. Create markdown files
# For each page.pdf, create page.md with:
cat > page1.md << 'EOF'
---
primary_language: en
is_rotation_valid: True
rotation_correction: 0
is_table: False
is_diagram: False
---
Your manually transcribed text here...
EOF
```

---

## Part 2: LoRA Fine-tuning (Recommended) ?

### Step 1: Create Config File

```bash
# Save as my_finetuning_config.yaml
cat > my_finetuning_config.yaml << 'EOF'
# Project metadata
project_name: my-olmocr-finetuning
run_name: my-custom-olmocr-v1

# Model configuration
model:
  name: allenai/olmOCR-2-7B-1025-FP8  # Start from latest model
  trust_remote_code: true
  torch_dtype: bfloat16
  use_flash_attention: true
  attn_implementation: flash_attention_2
  
  # LoRA settings
  use_lora: true
  lora_rank: 8
  lora_alpha: 32
  lora_dropout: 0.1
  lora_target_modules:
    - q_proj
    - v_proj
    - k_proj
    - o_proj

# Dataset configuration
dataset:
  train:
    - name: my_training_data
      root_dir: /path/to/my_training_data/train  # 80% of data
      pipeline: &basic_pipeline
        - name: FrontMatterParser
          front_matter_class: PageResponse
        - name: FilterOutRotatedDocuments
        - name: ReformatLatexBoldItalic
        - name: DatasetTextRuleFilter
        - name: PDFRenderer
          target_longest_image_dim: 1288
        - name: RotationAugmentation
          probability: 0.02
        - name: NewYamlFinetuningPromptWithNoAnchoring
        - name: FrontMatterOutputFormat
        - name: InstructUserMessages
          prompt_first: true
        - name: Tokenizer
          masking_index: -100
          end_of_message_token: "<|im_end|>"

  eval:
    - name: my_eval_data
      root_dir: /path/to/my_training_data/eval  # 20% of data
      pipeline: *basic_pipeline

# Training configuration
training:
  output_dir: ./my_olmocr_checkpoints
  num_train_epochs: 3
  
  # Batch size (adjust for your GPU)
  per_device_train_batch_size: 1
  per_device_eval_batch_size: 1
  gradient_accumulation_steps: 32  # Effective batch = 32
  
  gradient_checkpointing: False
  collator_max_token_len: 8192
  
  # Learning rate
  learning_rate: 2e-5
  lr_scheduler_type: linear
  warmup_ratio: 0.1
  
  # Optimization
  optim: adamw_torch
  weight_decay: 0.01
  max_grad_norm: 1.0
  
  # Torch compile (faster training)
  torch_compile: true
  torch_compile_backend: inductor
  torch_compile_mode: default
  
  seed: 42
  data_seed: 43
  
  # Evaluation and checkpointing
  evaluation_strategy: steps
  eval_steps: 100
  save_strategy: steps
  save_steps: 100
  save_total_limit: 3
  metric_for_best_model: eval_my_eval_data_loss
  greater_is_better: false
  
  report_to: 
    - wandb  # Optional: remove if not using wandb

# Logging
logging:
  level: INFO
  log_steps: 10
EOF
```

### Step 2: Split Your Data

```bash
# Split into train (80%) and eval (20%)
mkdir -p /path/to/my_training_data/train
mkdir -p /path/to/my_training_data/eval

# Move ~80% of files to train/, 20% to eval/
# Or use a script:

python << 'SCRIPT'
import os
import random
import shutil
from pathlib import Path

source_dir = Path("./my_training_data")
train_dir = Path("./my_training_data/train")
eval_dir = Path("./my_training_data/eval")

train_dir.mkdir(exist_ok=True)
eval_dir.mkdir(exist_ok=True)

# Get all PDF files
pdf_files = list(source_dir.glob("*.pdf"))
random.shuffle(pdf_files)

# Split 80/20
split_idx = int(len(pdf_files) * 0.8)
train_files = pdf_files[:split_idx]
eval_files = pdf_files[split_idx:]

# Move files
for pdf in train_files:
    md_file = pdf.with_suffix('.md')
    shutil.move(str(pdf), str(train_dir / pdf.name))
    shutil.move(str(md_file), str(train_dir / md_file.name))

for pdf in eval_files:
    md_file = pdf.with_suffix('.md')
    shutil.move(str(pdf), str(eval_dir / pdf.name))
    shutil.move(str(md_file), str(eval_dir / md_file.name))

print(f"Train: {len(train_files)} files")
print(f"Eval: {len(eval_files)} files")
SCRIPT
```

### Step 3: Run Training

```bash
# Start training
python -m olmocr.train.train --config my_finetuning_config.yaml
```

**Training time:**
- ~100 samples: 30 minutes - 1 hour
- ~1000 samples: 3-6 hours
- ~10000 samples: 1-2 days

Monitor progress with wandb (if enabled) or check logs.

---

## Part 3: Prepare Your Trained Model

### Step 1: Merge LoRA Weights

After training completes, merge LoRA adapter back into base model:

```bash
# Find your best checkpoint
ls ./my_olmocr_checkpoints/

# Prepare checkpoint (merges LoRA, fixes configs)
python -m olmocr.train.prepare_checkpoint \
  ./my_olmocr_checkpoints/checkpoint-300 \
  ./my_olmocr_model
```

### Step 2: Quantize to FP8 (Recommended)

Makes model faster and smaller with minimal quality loss:

```bash
python -m olmocr.train.compress_checkpoint \
  --config olmocr/train/quantization_configs/qwen2_5vl_w8a8_fp8.yaml \
  ./my_olmocr_model \
  ./my_olmocr_model_FP8
```

**File sizes:**
- BF16 model: ~15GB
- FP8 model: ~7.5GB (2x smaller!)

---

## Part 4: Test Your Model

### Quick Test

```bash
# Download sample
curl -o test.pdf https://olmocr.allenai.org/papers/olmocr_3pg_sample.pdf

# Test with your model
python -m olmocr.pipeline ./test_workspace \
  --model ./my_olmocr_model_FP8 \
  --markdown \
  --pdfs test.pdf

# Check results
cat ./test_workspace/markdown/test.md
```

### Run Benchmark

```bash
# Install benchmark dependencies
pip install olmocr[bench]

# Download olmOCR-bench
huggingface-cli download --repo-type dataset \
  --resume-download allenai/olmOCR-bench \
  --local-dir ./olmOCR-bench

# Run benchmark
python -m olmocr.bench.benchmark \
  --model ./my_olmocr_model_FP8 \
  --data ./olmOCR-bench/bench_data \
  --output ./benchmark_results
```

---

## Part 5: Use Your Custom Model

### Local Inference

```bash
# Use like normal olmocr, but specify your model
python -m olmocr.pipeline ./my_workspace \
  --model ./my_olmocr_model_FP8 \
  --markdown \
  --pdfs /path/to/your/documents/*.pdf
```

### Deploy as API

```bash
# Start vLLM server with your model
vllm serve ./my_olmocr_model_FP8 \
  --served-model-name my-custom-olmocr \
  --max-model-len 16384

# Use with olmocr
python -m olmocr.pipeline ./workspace \
  --server http://localhost:8000/v1 \
  --model my-custom-olmocr \
  --markdown \
  --pdfs documents/*.pdf
```

---

## Common Issues & Solutions

### Issue: Out of Memory During Training

**Solution 1**: Reduce batch size
```yaml
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 64  # Increase this
```

**Solution 2**: Enable gradient checkpointing
```yaml
training:
  gradient_checkpointing: true
```

**Solution 3**: Reduce max token length
```yaml
training:
  collator_max_token_len: 4096  # Down from 8192
```

### Issue: Training is Slow

**Solution 1**: Enable torch compile
```yaml
training:
  torch_compile: true
```

**Solution 2**: Use smaller LoRA rank
```yaml
model:
  lora_rank: 4  # Down from 8
```

### Issue: Model Overfits (Train loss low, eval loss high)

**Solution 1**: Reduce epochs
```yaml
training:
  num_train_epochs: 1  # Down from 3
```

**Solution 2**: Add more data augmentation
```yaml
dataset:
  train:
    - pipeline:
      - name: RotationAugmentation
        probability: 0.05  # Up from 0.02
```

**Solution 3**: Increase LoRA dropout
```yaml
model:
  lora_dropout: 0.2  # Up from 0.1
```

---

## Cost Estimate

**LoRA Fine-tuning on Azure NC24ads_A100_v4:**

| Dataset Size | Training Time | Cost |
|-------------|---------------|------|
| 100 pages | 30-60 min | ~$2-4 |
| 1000 pages | 3-6 hours | ~$11-22 |
| 10000 pages | 1-2 days | ~$88-176 |

**Plus one-time setup costs:**
- Preparing data: Variable (depends on if you use olmocr to generate)
- Testing/validation: ~$10-20

---

## Best Practices

1. **Start Small**: Train on 100-200 pages first to validate pipeline
2. **Clean Data**: Better to have 500 perfect examples than 5000 noisy ones
3. **Monitor Overfitting**: Watch eval loss - if it goes up while train loss goes down, stop early
4. **Save Checkpoints**: Keep multiple checkpoints in case later ones overfit
5. **Test Often**: Validate on real documents frequently during training
6. **Version Control**: Keep your config files in git

---

## Quick Reference Commands

```bash
# 1. Prepare data from olmocr output
python -m olmocr.data.prepare_workspace ./workspace ./train_data

# 2. Train with LoRA
python -m olmocr.train.train --config my_config.yaml

# 3. Prepare checkpoint
python -m olmocr.train.prepare_checkpoint ./checkpoints/checkpoint-300 ./my_model

# 4. Quantize to FP8
python -m olmocr.train.compress_checkpoint \
  --config olmocr/train/quantization_configs/qwen2_5vl_w8a8_fp8.yaml \
  ./my_model ./my_model_FP8

# 5. Test
python -m olmocr.pipeline ./test --model ./my_model_FP8 --markdown --pdfs test.pdf
```

---

## Next Steps

1. **Collect Data**: Get 100-1000 pages of your document type
2. **Process with olmocr**: Use cloud inference to get initial transcriptions
3. **Review & Fix**: Manually correct errors in the output
4. **Train**: Run LoRA fine-tuning (3-6 hours)
5. **Evaluate**: Test on held-out documents
6. **Deploy**: Use your custom model for production

**Need help?** Check the [official training README](https://github.com/allenai/olmocr/blob/main/olmocr/train/README.md)

Good luck with your training! ??
