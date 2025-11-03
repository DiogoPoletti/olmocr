# Quick Commands After SSH

## ?? CRITICAL: T4 CANNOT RUN FP8 MODELS!

**Your Tesla T4 has compute capability 7.5, but FP8 requires 8.0+**

? Local inference with default model = **WON'T WORK**
? Cloud inference = **WORKS PERFECTLY** (and is cheaper!)

---

## Option 1: Cloud Inference (RECOMMENDED - Works immediately!)

### Quick Setup:
```bash
# Install olmocr (no GPU packages needed!)
conda create -n olmocr python=3.11 -y
conda activate olmocr
pip install olmocr

# Download sample
mkdir -p ~/test && cd ~/test
curl -o sample.pdf https://olmocr.allenai.org/papers/olmocr_3pg_sample.pdf
```

### Choose a Provider and Get API Key:

**DeepInfra (Easiest - Sign up with Google/GitHub):**
1. Go to https://deepinfra.com/
2. Sign up ? Get API key
3. Run:
```bash
python -m olmocr.pipeline ./workspace --markdown \
  --server https://api.deepinfra.com/v1/openai \
  --api_key YOUR_API_KEY \
  --model allenai/olmOCR-2-7B-1025 \
  --pdfs sample.pdf
```

**Cirrascale (Cheapest - $0.07/M tokens):**
```bash
python -m olmocr.pipeline ./workspace --markdown \
  --server https://ai2endpoints.cirrascale.ai/api \
  --api_key YOUR_API_KEY \
  --model olmOCR-2-7B-1025 \
  --pdfs sample.pdf
```

**Cost**: ~$0.0003/page. 1000 pages = $0.30. Way cheaper than your VM!

---

## Option 2: Manual Setup to Understand the Issue (Will fail, but educational)

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

### 6. Try to Run (This will FAIL with FP8 error)
```bash
python -m olmocr.pipeline ./workspace_test --markdown --pdfs sample.pdf \
  --gpu-memory-utilization 0.70 \
  --max_model_len 10240
```

**This will fail with**: `RuntimeError: Quantization scheme is not supported for the current GPU. Min capability: 80. Current capability: 75.`

This is expected! The T4 cannot run FP8 models. Use cloud inference instead (Option 1 above).

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
