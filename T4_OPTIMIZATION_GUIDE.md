# olmocr GPU Memory Optimization Guide for Tesla T4 16GB

## Problem
The Tesla T4 16GB GPU in NC4as_T4_v3 is just barely meeting the minimum requirement (15GB) for olmocr, causing potential out-of-memory issues.

## Quick Fix Commands

### Option 1: Reduced Memory Settings (Try First)
```bash
# Conservative settings - should work on 16GB T4
python -m olmocr.pipeline ./localworkspace --markdown --pdfs your_file.pdf \
  --gpu-memory-utilization 0.70 \
  --max_model_len 10240

# If that works, try slightly higher settings for better performance
python -m olmocr.pipeline ./localworkspace --markdown --pdfs your_file.pdf \
  --gpu-memory-utilization 0.75 \
  --max_model_len 12288
```

### Option 2: Use External Inference (Recommended for Production)
```bash
# Cirrascale - Cheapest option ($0.07/M input tokens, $0.15/M output tokens)
python -m olmocr.pipeline ./localworkspace \
  --server https://ai2endpoints.cirrascale.ai/api \
  --api_key YOUR_API_KEY \
  --model olmOCR-2-7B-1025 \
  --markdown \
  --pdfs your_files/*.pdf

# DeepInfra - Easy to use ($0.09/M input tokens, $0.19/M output tokens)
python -m olmocr.pipeline ./localworkspace \
  --server https://api.deepinfra.com/v1/openai \
  --api_key YOUR_API_KEY \
  --model allenai/olmOCR-2-7B-1025 \
  --markdown \
  --pdfs your_files/*.pdf

# Parasail - Serverless ($0.10/M input tokens, $0.20/M output tokens)
python -m olmocr.pipeline ./localworkspace \
  --server https://api.parasail.io/v1 \
  --api_key YOUR_API_KEY \
  --model allenai/olmOCR-2-7B-1025 \
  --markdown \
  --pdfs your_files/*.pdf
```

## Detailed Parameters Explanation

### --gpu-memory-utilization
- **Default**: ~0.90 (90% of VRAM)
- **Recommended for T4 16GB**: 0.70-0.75
- Controls how much VRAM vLLM pre-allocates for KV-cache
- Lower = more conservative, less likely to crash

### --max_model_len
- **Default**: 16384 tokens
- **Recommended for T4 16GB**: 10240-12288 tokens
- Upper bound for KV-cache allocation
- Lower = less memory usage, but handles shorter documents per batch

## Cost Analysis: Local vs Cloud

### Local T4 16GB (NC4as_T4_v3)
- **Cost**: ~$0.526/hour (varies by region)
- **Problems**: Memory constraints, may crash
- **Best for**: Development, small batches

### Cloud Inference Providers
- **Cost**: $0.07-$0.10 per 1M input tokens
- **Typical document**: ~5,000 tokens = $0.0003-$0.0005 per page
- **1000 pages**: $0.30-$0.50
- **Best for**: Production, large batches

**Break-even point**: If processing more than ~1-2 hours worth of documents, cloud providers are cheaper and more reliable.

## VM Upgrade Options (If Needed)

If you must run locally and need more memory:

1. **NC8as_T4_v3** (2x T4 16GB) - Can use `--tensor-parallel-size 2`
2. **NC12s_v3** (V100 24GB) - More headroom
3. **NC24ads_A100_v4** (A100 40GB) - Tested configuration, best performance

## Monitoring Memory Usage

Check if you're hitting limits:
```bash
# In another terminal while running olmocr
watch -n 1 nvidia-smi
```

Look for:
- Memory usage approaching 16GB
- "Out of memory" errors in logs
- "KV cache is larger than available memory" errors

## Example Workflow for T4 16GB

```bash
# 1. Start with conservative settings
python -m olmocr.pipeline ./workspace1 --markdown \
  --gpu-memory-utilization 0.70 \
  --max_model_len 10240 \
  --pdfs test_document.pdf

# 2. If successful, process a few documents
python -m olmocr.pipeline ./workspace1 --markdown \
  --gpu-memory-utilization 0.70 \
  --max_model_len 10240 \
  --pdfs documents/*.pdf

# 3. Monitor with nvidia-smi to see actual usage
# 4. Adjust parameters up/down based on results

# 5. For large batches, consider switching to cloud:
python -m olmocr.pipeline ./workspace_prod \
  --server https://ai2endpoints.cirrascale.ai/api \
  --api_key YOUR_KEY \
  --model olmOCR-2-7B-1025 \
  --markdown \
  --pdfs large_batch/*.pdf
```

## Additional Tips

1. **Process PDFs in smaller batches** if memory issues persist
2. **Use `--workers 1`** to limit concurrent processing
3. **Monitor system memory** too - ensure you have enough RAM (not just VRAM)
4. **Use the FP8 model** (default) - it's already optimized for lower memory
5. **Close other GPU applications** before running olmocr
6. **Consider Docker** for cleaner environment: `docker pull alleninstituteforai/olmocr:latest`

## Troubleshooting

### Error: "CUDA out of memory"
Solution: Reduce `--gpu-memory-utilization` to 0.65 or lower

### Error: "KV cache is larger than available memory"
Solution: Reduce `--max_model_len` to 8192 or lower

### Error: vLLM won't start
Solution: Try both parameters together:
```bash
--gpu-memory-utilization 0.65 --max_model_len 8192
```

### Documents are too long
If your documents exceed the token limit, they'll be automatically chunked. Lower `--max_model_len` might work but could affect quality on very long documents.

## Conclusion

For a 16GB T4, the recommended approach is:
1. **Development/Testing**: Use local with `--gpu-memory-utilization 0.70 --max_model_len 10240`
2. **Production/Scale**: Use external providers like Cirrascale for better cost and reliability
3. **Heavy Usage**: Consider upgrading VM to NC12s_v3 or NC24ads_A100_v4
