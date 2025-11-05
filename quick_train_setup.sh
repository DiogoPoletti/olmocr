#!/bin/bash
# Quick setup script for olmocr training on your data

set -e

echo "=========================================="
echo "olmocr Training Quick Setup"
echo "=========================================="
echo ""

# Check GPU
if ! nvidia-smi &> /dev/null; then
    echo "ERROR: No GPU detected!"
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)

echo "Detected GPU: $GPU_NAME"
echo "GPU Memory: ${GPU_MEM}MB"
echo ""

# Check minimum requirements
if [ "$GPU_MEM" -lt 40000 ]; then
    echo "WARNING: Less than 40GB GPU memory detected."
    echo "Training may fail or be very slow."
    echo "Recommended: A100 40GB or better"
    echo ""
fi

# Install dependencies
echo "Installing training dependencies..."
pip install olmocr[train] -q
pip install transformers==4.52.4 -q
pip install flash-attn>=2.8.0.post2 --no-build-isolation -q

echo "? Dependencies installed"
echo ""

# Create directory structure
echo "Creating directory structure..."
mkdir -p ~/olmocr_training/{data/raw,data/train,data/eval,configs,checkpoints,models}
cd ~/olmocr_training

echo "? Directories created at ~/olmocr_training"
echo ""

# Create example config
echo "Creating example training config..."
cat > configs/my_finetuning.yaml << 'EOF'
# olmocr LoRA Fine-tuning Configuration
project_name: my-olmocr-training
run_name: finetuning-v1

# Model settings
model:
  name: allenai/olmOCR-2-7B-1025-FP8
  trust_remote_code: true
  torch_dtype: bfloat16
  use_flash_attention: true
  attn_implementation: flash_attention_2
  
  # LoRA configuration
  use_lora: true
  lora_rank: 8
  lora_alpha: 32
  lora_dropout: 0.1
  lora_target_modules:
    - q_proj
    - v_proj
    - k_proj
    - o_proj

# Dataset paths (UPDATE THESE!)
dataset:
  train:
    - name: training_data
      root_dir: ~/olmocr_training/data/train
      pipeline: &pipeline
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
    - name: eval_data
      root_dir: ~/olmocr_training/data/eval
      pipeline: *pipeline

# Training settings
training:
  output_dir: ~/olmocr_training/checkpoints
  num_train_epochs: 3
  
  per_device_train_batch_size: 1
  per_device_eval_batch_size: 1
  gradient_accumulation_steps: 32
  
  gradient_checkpointing: false
  collator_max_token_len: 8192
  
  learning_rate: 2e-5
  lr_scheduler_type: linear
  warmup_ratio: 0.1
  
  optim: adamw_torch
  weight_decay: 0.01
  max_grad_norm: 1.0
  
  torch_compile: true
  torch_compile_backend: inductor
  torch_compile_mode: default
  
  seed: 42
  data_seed: 43
  
  evaluation_strategy: steps
  eval_steps: 100
  save_strategy: steps
  save_steps: 100
  save_total_limit: 3
  
  report_to: []  # Add 'wandb' if you want logging

logging:
  level: INFO
  log_steps: 10
EOF

echo "? Config created at configs/my_finetuning.yaml"
echo ""

# Create data preparation helper
cat > prepare_my_data.py << 'PYTHON'
#!/usr/bin/env python3
"""Helper script to split training data into train/eval sets"""

import random
import shutil
from pathlib import Path

def split_data(source_dir, train_ratio=0.8):
    source = Path(source_dir)
    train_dir = Path("data/train")
    eval_dir = Path("data/eval")
    
    train_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all PDF files
    pdf_files = list(source.glob("*.pdf"))
    if not pdf_files:
        print(f"ERROR: No PDF files found in {source_dir}")
        return
    
    random.shuffle(pdf_files)
    
    # Split
    split_idx = int(len(pdf_files) * train_ratio)
    train_files = pdf_files[:split_idx]
    eval_files = pdf_files[split_idx:]
    
    # Copy files
    for pdf in train_files:
        md_file = pdf.with_suffix('.md')
        if md_file.exists():
            shutil.copy(str(pdf), str(train_dir / pdf.name))
            shutil.copy(str(md_file), str(train_dir / md_file.name))
    
    for pdf in eval_files:
        md_file = pdf.with_suffix('.md')
        if md_file.exists():
            shutil.copy(str(pdf), str(eval_dir / pdf.name))
            shutil.copy(str(md_file), str(eval_dir / md_file.name))
    
    print(f"? Split complete:")
    print(f"  Train: {len(train_files)} files ? data/train/")
    print(f"  Eval: {len(eval_files)} files ? data/eval/")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python prepare_my_data.py <source_directory>")
        print("Example: python prepare_my_data.py data/raw")
        sys.exit(1)
    
    split_data(sys.argv[1])
PYTHON

chmod +x prepare_my_data.py

echo "? Data preparation script created"
echo ""

# Create training helper script
cat > start_training.sh << 'BASH'
#!/bin/bash
# Start olmocr training

set -e

echo "Starting olmocr training..."
echo ""

# Check if data exists
if [ ! -d "data/train" ] || [ -z "$(ls -A data/train/*.pdf 2>/dev/null)" ]; then
    echo "ERROR: No training data found in data/train/"
    echo ""
    echo "Please prepare your data first:"
    echo "  1. Put your PDF+MD files in data/raw/"
    echo "  2. Run: python prepare_my_data.py data/raw"
    exit 1
fi

# Count files
TRAIN_COUNT=$(ls data/train/*.pdf 2>/dev/null | wc -l)
EVAL_COUNT=$(ls data/eval/*.pdf 2>/dev/null | wc -l)

echo "Training data: $TRAIN_COUNT files"
echo "Eval data: $EVAL_COUNT files"
echo ""

if [ "$TRAIN_COUNT" -lt 10 ]; then
    echo "WARNING: Very small training set (<10 files)"
    echo "Consider adding more data for better results"
    echo ""
fi

# Start training
python -m olmocr.train.train --config configs/my_finetuning.yaml

echo ""
echo "Training complete!"
echo "Checkpoints saved in: checkpoints/"
echo ""
echo "Next steps:"
echo "  1. Prepare checkpoint: python -m olmocr.train.prepare_checkpoint checkpoints/checkpoint-XXX models/my_model"
echo "  2. Quantize (optional): python -m olmocr.train.compress_checkpoint --config ~/olmocr_training/quantization_config.yaml models/my_model models/my_model_FP8"
echo "  3. Test: python -m olmocr.pipeline ./test --model models/my_model_FP8 --markdown --pdfs test.pdf"
BASH

chmod +x start_training.sh

echo "? Training script created"
echo ""

# Create README
cat > README.txt << 'TXT'
olmocr Training Quick Setup
============================

Directory Structure:
  data/raw/     - Put your original PDF+MD files here
  data/train/   - Training data (auto-generated)
  data/eval/    - Evaluation data (auto-generated)
  configs/      - Training configurations
  checkpoints/  - Saved checkpoints during training
  models/       - Final prepared models

Quick Start:
============

Option 1: Use olmocr to Generate Training Data
-----------------------------------------------
1. Convert your PDFs with olmocr:
   python -m olmocr.pipeline ./workspace --pdfs /path/to/*.pdf

2. Review and fix the output in ./workspace/

3. Convert to training format:
   python -m olmocr.data.prepare_workspace ./workspace ./data/raw

4. Split into train/eval:
   python prepare_my_data.py data/raw

5. Start training:
   ./start_training.sh

Option 2: Manual Training Data
-------------------------------
1. Create PDF+MD pairs in data/raw/:
   - Each PDF must be a single page
   - Each PDF needs a matching .md file
   - MD files need YAML front matter (see example below)

2. Split data:
   python prepare_my_data.py data/raw

3. Start training:
   ./start_training.sh

Example Markdown Format:
------------------------
---
primary_language: en
is_rotation_valid: True
rotation_correction: 0
is_table: False
is_diagram: False
---
Your document text here...

After Training:
===============
1. Prepare checkpoint:
   python -m olmocr.train.prepare_checkpoint \
     checkpoints/checkpoint-300 \
     models/my_model

2. Quantize to FP8 (optional but recommended):
   python -m olmocr.train.compress_checkpoint \
     --config olmocr/train/quantization_configs/qwen2_5vl_w8a8_fp8.yaml \
     models/my_model \
     models/my_model_FP8

3. Test your model:
   python -m olmocr.pipeline ./test \
     --model models/my_model_FP8 \
     --markdown \
     --pdfs sample.pdf

Need Help?
==========
See TRAINING_GUIDE.md for detailed instructions
TXT

echo "=========================================="
echo "? Setup Complete!"
echo "=========================================="
echo ""
echo "Your training environment is ready at: ~/olmocr_training"
echo ""
echo "Next steps:"
echo ""
echo "1. Prepare your training data:"
echo "   Option A: Use olmocr to generate from your PDFs"
echo "   Option B: Manually create PDF+MD pairs"
echo ""
echo "2. Place data in: ~/olmocr_training/data/raw/"
echo ""
echo "3. Split the data:"
echo "   cd ~/olmocr_training"
echo "   python prepare_my_data.py data/raw"
echo ""
echo "4. Start training:"
echo "   ./start_training.sh"
echo ""
echo "See ~/olmocr_training/README.txt for detailed instructions"
echo ""
echo "Good luck with your training! ??"
