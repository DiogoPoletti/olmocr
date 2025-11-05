# CRITICAL: Tesla T4 Cannot Run FP8 Models

## The Problem

```
RuntimeError: Quantization scheme is not supported for the current GPU. 
Min capability: 80. Current capability: 75.
```

**Tesla T4 has compute capability 7.5, but FP8 quantization requires 8.0+** (Ampere or newer GPUs like A100, H100, RTX 4090, L40S).

The default olmocr model `allenai/olmOCR-2-7B-1025-FP8` uses FP8 quantization and **will NOT work on T4**.

## Solutions (Ranked by Best to Worst)

### ? Solution 1: Use Cloud Inference (HIGHLY RECOMMENDED)

**This is the easiest and most cost-effective solution.** You don't need to change anything on your VM.

#### Cirrascale (Cheapest - $0.07/M tokens)
```bash
# 1. Sign up at https://ai2endpoints.cirrascale.ai/
# 2. Get your API key
# 3. Run:

python -m olmocr.pipeline ./my_workspace --markdown \
  --server https://ai2endpoints.cirrascale.ai/api \
  --api_key YOUR_API_KEY \
  --model olmOCR-2-7B-1025 \
  --pdfs /path/to/your/*.pdf
```

#### DeepInfra ($0.09/M tokens)
```bash
# 1. Sign up at https://deepinfra.com/
# 2. Get your API key
# 3. Run:

python -m olmocr.pipeline ./my_workspace --markdown \
  --server https://api.deepinfra.com/v1/openai \
  --api_key YOUR_API_KEY \
  --model allenai/olmOCR-2-7B-1025 \
  --pdfs /path/to/your/*.pdf
```

#### Parasail ($0.10/M tokens)
```bash
# 1. Sign up at https://www.saas.parasail.io/
# 2. Get your API key
# 3. Run:

python -m olmocr.pipeline ./my_workspace --markdown \
  --server https://api.parasail.io/v1 \
  --api_key YOUR_API_KEY \
  --model allenai/olmOCR-2-7B-1025 \
  --pdfs /path/to/your/*.pdf
```

**Cost**: ~$0.0003-$0.0005 per page. For 1000 pages = $0.30-$0.50. Way cheaper than running your T4 VM!

---

### ?? Solution 2: Use BF16/FP16 Model Locally (If Available)

Check if a non-quantized version exists on HuggingFace:

```bash
# Try using the base model without FP8 suffix
python -m olmocr.pipeline ./my_workspace --markdown \
  --model allenai/olmOCR-2-7B-1025 \
  --pdfs sample.pdf \
  --gpu-memory-utilization 0.70 \
  --max_model_len 10240
```

**Warning**: This will use ~2x more GPU memory than FP8. With only 16GB on T4, this might still fail or be very slow.

---

### ? Solution 3: Upgrade Your VM (Most Expensive)

If you absolutely need local inference and can't use cloud:

**Minimum that works with FP8:**
- **NC6s_v3** - V100 16GB (~$1.23/hr) - Still tight, compute capability 7.0 (FP8 won't work!)
- **NC8as_T4_v3** - 2? T4 16GB (~$1.05/hr) - Still won't work with FP8!
- **NC24ads_A100_v4** - A100 80GB (~$3.67/hr) - Works perfectly, tested by olmocr team

**Actually, you need Ampere or newer (A100, A10, L40S, H100, RTX 4000 series) for FP8!**

Most cost-effective options that support FP8:
- **NC6s_v4** - A10 24GB
- **NC24ads_A100_v4** - A100 80GB
- **ND96asr_v4** - 8? A100 (overkill for single user)

---

## Why Cloud Inference is Better

| Aspect | Your T4 VM | Cloud Inference |
|--------|-----------|-----------------|
| **Setup** | Complex, compatibility issues | Just API key |
| **Cost for 1000 pages** | ~$0.50+ (1 hour runtime) | $0.30-$0.50 total |
| **Cost for 10000 pages** | ~$5+ (10 hours) | $3-$5 total |
| **Maintenance** | You manage everything | Provider handles it |
| **Upgrades** | Requires VM change | Automatic |
| **Scalability** | Limited to 1 GPU | Unlimited |
| **FP8 Support** | ? NO | ? YES |

---

## Immediate Action

**Stop trying to run locally. Use cloud inference instead:**

```bash
# Quick test with DeepInfra (easy signup)
# 1. Go to https://deepinfra.com/
# 2. Sign up with Google/GitHub
# 3. Copy your API key from dashboard
# 4. Run:

curl -o ~/sample.pdf https://olmocr.allenai.org/papers/olmocr_3pg_sample.pdf

python -m olmocr.pipeline ./test_cloud --markdown \
  --server https://api.deepinfra.com/v1/openai \
  --api_key YOUR_API_KEY_HERE \
  --model allenai/olmOCR-2-7B-1025 \
  --pdfs ~/sample.pdf

# Check results
cat test_cloud/markdown/sample.md
```

This should work immediately without any GPU issues!

---

## Understanding the Error

The error occurs because:

1. **Tesla T4** = Turing architecture = Compute Capability 7.5
2. **FP8 quantization** = Requires Ampere or newer = Compute Capability 8.0+
3. **olmocr default model** = Uses FP8 = Won't load on T4

It's a fundamental hardware limitation. You can't fix this by adjusting memory settings or parameters.

---

## FAQ

**Q: Can I use a different quantization like INT8?**
A: The olmocr team only provides FP8 quantized models. You'd need to quantize yourself (not recommended).

**Q: Can I use the base Qwen2.5-VL model?**
A: Technically yes, but it's not trained for OCR and won't work well. Not worth it.

**Q: What if I need to process offline/air-gapped?**
A: You need to upgrade to an A100 VM or get physical hardware with Ampere+ GPUs.

**Q: Is there a free option?**
A: DeepInfra gives $25 free credit. Parasail has a free tier. Both enough to test.

**Q: How long does cloud inference take?**
A: Similar speed to local. ~10-30 seconds per page depending on complexity.

---

## Bottom Line

**For Tesla T4 users: Use cloud inference. Don't fight the hardware.**

Sign up for one of these providers (takes 2 minutes):
- Cirrascale: https://ai2endpoints.cirrascale.ai/
- DeepInfra: https://deepinfra.com/
- Parasail: https://www.saas.parasail.io/

Then use the commands above. Problem solved! ??
