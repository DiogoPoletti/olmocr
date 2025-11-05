# Quick Start for olmocr on Tesla T4 16GB (NC4as_T4_v3)

## TL;DR - Commands That Should Work

### Option A: Local Processing (Reduced Memory)
```bash
# For most documents - recommended starting point
python -m olmocr.pipeline ./localworkspace --markdown --pdfs your_file.pdf \
  --gpu-memory-utilization 0.70 \
  --max_model_len 10240

# If that fails, use even more conservative settings
python -m olmocr.pipeline ./localworkspace --markdown --pdfs your_file.pdf \
  --gpu-memory-utilization 0.65 \
  --max_model_len 8192
```

### Option B: Use Cloud Inference (Easier & Cheaper)
```bash
# Sign up at https://ai2endpoints.cirrascale.ai/ and get API key
python -m olmocr.pipeline ./localworkspace --markdown --pdfs your_files/*.pdf \
  --server https://ai2endpoints.cirrascale.ai/api \
  --api_key YOUR_API_KEY \
  --model olmOCR-2-7B-1025
```

**Cost**: ~$0.0003-$0.0005 per page (much cheaper than running your VM if processing large batches)

## Test Your Configuration

Run the included test script:
```bash
./test_t4_memory.sh
```

This will automatically test different memory configurations and recommend optimal settings for your GPU.

## Need Help?

See `T4_OPTIMIZATION_GUIDE.md` for detailed explanations and troubleshooting.

## External Inference Providers

| Provider | Input Cost | Output Cost | Sign Up Link |
|----------|-----------|------------|--------------|
| Cirrascale | $0.07/M | $0.15/M | https://ai2endpoints.cirrascale.ai/ |
| DeepInfra | $0.09/M | $0.19/M | https://deepinfra.com/ |
| Parasail | $0.10/M | $0.20/M | https://www.saas.parasail.io/ |

All are cheaper than running your VM for anything more than small test batches!
