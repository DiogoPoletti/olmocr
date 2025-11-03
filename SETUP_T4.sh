#!/bin/bash
# Complete setup script for olmocr on Tesla T4 16GB VM
# Run this after SSH'ing into your VM

set -e

echo "=========================================="
echo "olmocr Setup for Tesla T4 16GB"
echo "=========================================="
echo ""

# Step 1: Check GPU
echo "Step 1: Checking GPU..."
if ! nvidia-smi &> /dev/null; then
    echo "ERROR: No GPU detected. Make sure you're on the right VM."
    exit 1
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "? GPU detected"
echo ""

# Step 2: Check/Install system dependencies
echo "Step 2: Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y poppler-utils ttf-mscorefonts-installer msttcorefonts \
    fonts-crosextra-caladea fonts-crosextra-carlito gsfonts lcdf-typetools curl
echo "? System dependencies installed"
echo ""

# Step 3: Setup Python environment
echo "Step 3: Setting up Python environment..."
if ! command -v conda &> /dev/null; then
    echo "Conda not found. Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p $HOME/miniconda3
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    conda init
    echo "? Miniconda installed. Please run 'source ~/.bashrc' and run this script again."
    exit 0
fi

# Create/activate conda environment
if ! conda env list | grep -q "^olmocr "; then
    echo "Creating olmocr environment..."
    conda create -n olmocr python=3.11 -y
fi

echo "? Conda environment ready"
echo ""

# Step 4: Activate environment and install olmocr
echo "Step 4: Installing olmocr..."
echo "Activating olmocr environment..."

# Source conda
eval "$(conda shell.bash hook)"
conda activate olmocr

# Install olmocr with GPU support
pip install olmocr[gpu] --extra-index-url https://download.pytorch.org/whl/cu128

# Optional but recommended: Install flash infer for faster inference
echo "Installing flash-infer (optional, for faster inference)..."
pip install https://download.pytorch.org/whl/cu128/flashinfer/flashinfer_python-0.2.5%2Bcu128torch2.7-cp311-abi3-linux_x86_64.whl || echo "Flash-infer install failed (optional), continuing..."

echo "? olmocr installed"
echo ""

# Step 5: Download test PDF
echo "Step 5: Downloading test PDF..."
mkdir -p ~/olmocr_test
cd ~/olmocr_test
curl -o sample.pdf https://olmocr.allenai.org/papers/olmocr_3pg_sample.pdf
echo "? Test PDF downloaded"
echo ""

# Step 6: Test with optimized settings
echo "Step 6: Testing olmocr with T4-optimized settings..."
echo "This may take a few minutes on first run (downloading model)..."
echo ""

python -m olmocr.pipeline ./workspace_test --markdown --pdfs sample.pdf \
    --gpu-memory-utilization 0.70 \
    --max_model_len 10240

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "? SUCCESS! olmocr is working on your T4!"
    echo "=========================================="
    echo ""
    echo "Your results are in: ~/olmocr_test/workspace_test/markdown/sample.md"
    echo ""
    echo "View the result:"
    echo "  cat ~/olmocr_test/workspace_test/markdown/sample.md"
    echo ""
    echo "To process your own PDFs:"
    echo "  conda activate olmocr"
    echo "  python -m olmocr.pipeline ./my_workspace --markdown --pdfs /path/to/your/*.pdf \\"
    echo "    --gpu-memory-utilization 0.70 --max_model_len 10240"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "? Test failed. Your T4 might need more conservative settings."
    echo "=========================================="
    echo ""
    echo "Try this more conservative command:"
    echo "  python -m olmocr.pipeline ./workspace_test --markdown --pdfs sample.pdf \\"
    echo "    --gpu-memory-utilization 0.65 --max_model_len 8192"
    echo ""
    echo "Or consider using cloud inference (cheaper and easier):"
    echo "  python -m olmocr.pipeline ./workspace --markdown --pdfs sample.pdf \\"
    echo "    --server https://ai2endpoints.cirrascale.ai/api \\"
    echo "    --api_key YOUR_API_KEY \\"
    echo "    --model olmOCR-2-7B-1025"
fi
