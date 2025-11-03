#!/bin/bash
# Test script for olmocr on Tesla T4 16GB
# This script tests different memory configurations to find optimal settings

set -e

WORKSPACE_DIR="./test_workspace"
TEST_PDF="tests/gnarly_pdfs/arxiv-multicolumn.pdf"  # Use a test PDF from the repo

echo "=========================================="
echo "olmocr Memory Configuration Tester"
echo "For Tesla T4 16GB GPU"
echo "=========================================="
echo ""

# Check if running on GPU
if ! nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Are you running on a GPU instance?"
    exit 1
fi

# Show GPU info
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Test if test PDF exists, if not use a sample
if [ ! -f "$TEST_PDF" ]; then
    echo "Test PDF not found. Downloading sample..."
    mkdir -p test_pdfs
    curl -o test_pdfs/olmocr-sample.pdf https://olmocr.allenai.org/papers/olmocr_3pg_sample.pdf
    TEST_PDF="test_pdfs/olmocr-sample.pdf"
fi

echo "Test PDF: $TEST_PDF"
echo ""

# Configuration 1: Very Conservative (Most likely to succeed)
echo "=========================================="
echo "Test 1: Very Conservative Settings"
echo "GPU Memory: 0.65, Max Length: 8192"
echo "=========================================="

python -m olmocr.pipeline "$WORKSPACE_DIR/test1" \
    --markdown \
    --pdfs "$TEST_PDF" \
    --gpu-memory-utilization 0.65 \
    --max_model_len 8192

if [ $? -eq 0 ]; then
    echo "? Test 1 PASSED - Very conservative settings work!"
    echo ""
    
    # Configuration 2: Moderate
    echo "=========================================="
    echo "Test 2: Moderate Settings"
    echo "GPU Memory: 0.70, Max Length: 10240"
    echo "=========================================="
    
    python -m olmocr.pipeline "$WORKSPACE_DIR/test2" \
        --markdown \
        --pdfs "$TEST_PDF" \
        --gpu-memory-utilization 0.70 \
        --max_model_len 10240
    
    if [ $? -eq 0 ]; then
        echo "? Test 2 PASSED - Moderate settings work!"
        echo ""
        
        # Configuration 3: Optimal
        echo "=========================================="
        echo "Test 3: Optimal Settings"
        echo "GPU Memory: 0.75, Max Length: 12288"
        echo "=========================================="
        
        python -m olmocr.pipeline "$WORKSPACE_DIR/test3" \
            --markdown \
            --pdfs "$TEST_PDF" \
            --gpu-memory-utilization 0.75 \
            --max_model_len 12288
        
        if [ $? -eq 0 ]; then
            echo "? Test 3 PASSED - Optimal settings work!"
            echo ""
            echo "=========================================="
            echo "RECOMMENDATION:"
            echo "Use: --gpu-memory-utilization 0.75 --max_model_len 12288"
            echo "=========================================="
        else
            echo "? Test 3 FAILED - Optimal settings too aggressive"
            echo ""
            echo "=========================================="
            echo "RECOMMENDATION:"
            echo "Use: --gpu-memory-utilization 0.70 --max_model_len 10240"
            echo "=========================================="
        fi
    else
        echo "? Test 2 FAILED - Moderate settings too aggressive"
        echo ""
        echo "=========================================="
        echo "RECOMMENDATION:"
        echo "Use: --gpu-memory-utilization 0.65 --max_model_len 8192"
        echo "Consider using external inference providers instead."
        echo "=========================================="
    fi
else
    echo "? Test 1 FAILED - Even very conservative settings failed!"
    echo ""
    echo "=========================================="
    echo "RECOMMENDATION:"
    echo "Your T4 16GB may not have enough memory for local inference."
    echo "Options:"
    echo "1. Use external inference providers (recommended):"
    echo "   - Cirrascale: \$0.07/M tokens"
    echo "   - DeepInfra: \$0.09/M tokens"
    echo "   - Parasail: \$0.10/M tokens"
    echo ""
    echo "2. Upgrade to larger VM:"
    echo "   - NC12s_v3 (V100 24GB)"
    echo "   - NC24ads_A100_v4 (A100 40GB)"
    echo ""
    echo "See T4_OPTIMIZATION_GUIDE.md for details."
    echo "=========================================="
fi

echo ""
echo "Testing complete. Check results in $WORKSPACE_DIR/"
echo "Results are in markdown format in the markdown/ subdirectory."
