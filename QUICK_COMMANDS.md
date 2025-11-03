# Quick Commands After SSH

## Option 1: Automated Setup (Recommended)

Just run this one command - it does everything:

```bash
curl -sSL https://raw.githubusercontent.com/allenai/olmocr/main/SETUP_T4.sh | bash
```

*Note: If the script isn't in the repo yet, copy the commands below manually*

---

## Option 2: Manual Setup (Step by Step)

### 1. Check Your GPU
```bash
nvidia-smi
```
Should show "Tesla T4" with ~16GB memory.

### 2. Install System Dependencies
```bash
sudo apt-get update
sudo apt-get install -y poppler-utils ttf-mscorefonts-installer msttcorefonts \
  fonts-crosextra-caladea fonts-crosextra-carlito gsfonts lcdf-typetools
```

### 3. Create Conda Environment
```bash
# If you don't have conda, install it first:
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
~/miniconda3/bin/conda init
source ~/.bashrc

# Create environment
conda create -n olmocr python=3.11 -y
conda activate olmocr
```

### 4. Install olmocr
```bash
pip install olmocr[gpu] --extra-index-url https://download.pytorch.org/whl/cu128

# Optional but recommended - faster inference
pip install https://download.pytorch.org/whl/cu128/flashinfer/flashinfer_python-0.2.5%2Bcu128torch2.7-cp311-abi3-linux_x86_64.whl
```

### 5. Download Test PDF
```bash
mkdir -p ~/olmocr_test && cd ~/olmocr_test
curl -o sample.pdf https://olmocr.allenai.org/papers/olmocr_3pg_sample.pdf
```

### 6. Run First Test (T4-Optimized)
```bash
python -m olmocr.pipeline ./workspace_test --markdown --pdfs sample.pdf \
  --gpu-memory-utilization 0.70 \
  --max_model_len 10240
```

**First run will take 5-10 minutes** (downloading ~15GB model). Subsequent runs are fast.

### 7. View Results
```bash
cat workspace_test/markdown/sample.md
```

---

## Process Your Own PDFs

Once the test works:

```bash
conda activate olmocr

# Single PDF
python -m olmocr.pipeline ./my_workspace --markdown \
  --pdfs /path/to/your/document.pdf \
  --gpu-memory-utilization 0.70 \
  --max_model_len 10240

# Multiple PDFs
python -m olmocr.pipeline ./my_workspace --markdown \
  --pdfs /path/to/pdfs/*.pdf \
  --gpu-memory-utilization 0.70 \
  --max_model_len 10240
```

Results will be in: `./my_workspace/markdown/`

---

## If You Get "Out of Memory" Errors

Try more conservative settings:

```bash
python -m olmocr.pipeline ./workspace_test --markdown --pdfs sample.pdf \
  --gpu-memory-utilization 0.65 \
  --max_model_len 8192
```

Or use cloud inference instead (easier and often cheaper):

```bash
# 1. Sign up at https://ai2endpoints.cirrascale.ai/ (get API key)

# 2. Run without local GPU:
python -m olmocr.pipeline ./my_workspace --markdown \
  --server https://ai2endpoints.cirrascale.ai/api \
  --api_key YOUR_API_KEY_HERE \
  --model olmOCR-2-7B-1025 \
  --pdfs /path/to/pdfs/*.pdf
```

**Cloud cost**: ~$0.0003-$0.0005 per page (very cheap!)

---

## Monitoring GPU Usage

Open another SSH session and run:
```bash
watch -n 1 nvidia-smi
```

This shows real-time GPU memory usage while olmocr is running.

---

## Common Issues

### "pdftoppm not found"
```bash
sudo apt-get install poppler-utils
```

### "CUDA out of memory"
Lower the memory settings:
```bash
--gpu-memory-utilization 0.60 --max_model_len 8192
```

### "Cannot find conda"
```bash
source ~/.bashrc
conda activate olmocr
```

### Model download is slow
Be patient! First run downloads ~15GB. Grab a coffee ?

---

## Next Steps

- See `T4_OPTIMIZATION_GUIDE.md` for detailed tuning
- See `T4_QUICK_START.md` for more options
- Run `./test_t4_memory.sh` to find optimal settings automatically
