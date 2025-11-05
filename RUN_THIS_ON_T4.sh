#!/bin/bash
# WORKING SOLUTION for Tesla T4 16GB
# The T4 cannot run FP8 models due to compute capability limitations
# This script sets up cloud inference instead

set -e

echo "=========================================="
echo "olmocr Cloud Inference Setup for T4"
echo "=========================================="
echo ""
echo "??  IMPORTANT: Tesla T4 cannot run FP8 models!"
echo "   T4 has compute capability 7.5"
echo "   FP8 requires compute capability 8.0+ (A100, H100, RTX 4090, etc.)"
echo ""
echo "? SOLUTION: Use cloud inference (easier and cheaper anyway!)"
echo ""
echo "=========================================="
echo ""

# Check if they have conda
if ! command -v conda &> /dev/null; then
    echo "Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p $HOME/miniconda3
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    conda init
    source ~/.bashrc
fi

# Setup environment
eval "$(conda shell.bash hook)"

if ! conda env list | grep -q "^olmocr "; then
    echo "Creating olmocr environment..."
    conda create -n olmocr python=3.11 -y
fi

conda activate olmocr

# Install olmocr (no GPU dependencies needed for cloud inference!)
echo "Installing olmocr..."
pip install olmocr

# Download sample PDF
echo "Downloading sample PDF..."
mkdir -p ~/olmocr_cloud_test
cd ~/olmocr_cloud_test
curl -o sample.pdf https://olmocr.allenai.org/papers/olmocr_3pg_sample.pdf

echo ""
echo "=========================================="
echo "? Setup Complete!"
echo "=========================================="
echo ""
echo "Now you need to choose a cloud provider and get an API key:"
echo ""
echo "OPTION 1 - Cirrascale (Cheapest: \$0.07/M tokens)"
echo "  1. Sign up: https://ai2endpoints.cirrascale.ai/"
echo "  2. Get API key from dashboard"
echo "  3. Run:"
echo ""
echo "     python -m olmocr.pipeline ./workspace --markdown \\"
echo "       --server https://ai2endpoints.cirrascale.ai/api \\"
echo "       --api_key YOUR_API_KEY \\"
echo "       --model olmOCR-2-7B-1025 \\"
echo "       --pdfs sample.pdf"
echo ""
echo "OPTION 2 - DeepInfra (Easy signup: \$0.09/M tokens)"
echo "  1. Sign up: https://deepinfra.com/"
echo "  2. Get API key from dashboard"
echo "  3. Run:"
echo ""
echo "     python -m olmocr.pipeline ./workspace --markdown \\"
echo "       --server https://api.deepinfra.com/v1/openai \\"
echo "       --api_key YOUR_API_KEY \\"
echo "       --model allenai/olmOCR-2-7B-1025 \\"
echo "       --pdfs sample.pdf"
echo ""
echo "OPTION 3 - Parasail (Serverless: \$0.10/M tokens)"
echo "  1. Sign up: https://www.saas.parasail.io/"
echo "  2. Get API key"
echo "  3. Run:"
echo ""
echo "     python -m olmocr.pipeline ./workspace --markdown \\"
echo "       --server https://api.parasail.io/v1 \\"
echo "       --api_key YOUR_API_KEY \\"
echo "       --model allenai/olmOCR-2-7B-1025 \\"
echo "       --pdfs sample.pdf"
echo ""
echo "Cost: ~\$0.0003-\$0.0005 per page"
echo "Much cheaper than running your T4 VM! ??"
echo ""
echo "After running, check your results:"
echo "  cat workspace/markdown/sample.md"
echo ""
echo "=========================================="

# Save a quick reference file
cat > ~/olmocr_cloud_test/COMMANDS.txt << 'EOF'
# Quick Reference Commands for olmocr with Cloud Inference

# Activate environment
conda activate olmocr

# DeepInfra (easiest to get started)
python -m olmocr.pipeline ./workspace --markdown \
  --server https://api.deepinfra.com/v1/openai \
  --api_key YOUR_API_KEY \
  --model allenai/olmOCR-2-7B-1025 \
  --pdfs /path/to/your/*.pdf

# Cirrascale (cheapest)
python -m olmocr.pipeline ./workspace --markdown \
  --server https://ai2endpoints.cirrascale.ai/api \
  --api_key YOUR_API_KEY \
  --model olmOCR-2-7B-1025 \
  --pdfs /path/to/your/*.pdf

# Parasail (serverless)
python -m olmocr.pipeline ./workspace --markdown \
  --server https://api.parasail.io/v1 \
  --api_key YOUR_API_KEY \
  --model allenai/olmOCR-2-7B-1025 \
  --pdfs /path/to/your/*.pdf

# View results
cat workspace/markdown/your_file.md

# Process multiple files
python -m olmocr.pipeline ./workspace --markdown \
  --server https://api.deepinfra.com/v1/openai \
  --api_key YOUR_API_KEY \
  --model allenai/olmOCR-2-7B-1025 \
  --pdfs /path/to/folder/*.pdf
EOF

echo "Saved command reference to: ~/olmocr_cloud_test/COMMANDS.txt"
