# START HERE - Tesla T4 Users

## ?? YOUR T4 CANNOT RUN THE DEFAULT OLMOCR MODEL

The error you got is because:
- **Tesla T4** = Compute capability 7.5  
- **FP8 models** = Require compute capability 8.0+  
- **Default olmocr** = Uses FP8 = **Won't work on T4**

## ? The Fix: Use Cloud Inference (2 Minutes to Setup)

### Step 1: Install olmocr (No GPU needed!)
```bash
conda create -n olmocr python=3.11 -y
conda activate olmocr
pip install olmocr
```

### Step 2: Get a Free API Key

Pick ONE of these (all have free trials):

- **DeepInfra** (easiest): https://deepinfra.com/
- **Cirrascale** (cheapest): https://ai2endpoints.cirrascale.ai/
- **Parasail** (serverless): https://www.saas.parasail.io/

Sign up ? Copy your API key

### Step 3: Test It

```bash
# Download sample
mkdir -p ~/test && cd ~/test
curl -o sample.pdf https://olmocr.allenai.org/papers/olmocr_3pg_sample.pdf

# Run with DeepInfra (replace YOUR_API_KEY)
python -m olmocr.pipeline ./workspace --markdown \
  --server https://api.deepinfra.com/v1/openai \
  --api_key YOUR_API_KEY \
  --model allenai/olmOCR-2-7B-1025 \
  --pdfs sample.pdf

# Check results
cat workspace/markdown/sample.md
```

### Step 4: Process Your PDFs

```bash
python -m olmocr.pipeline ./my_workspace --markdown \
  --server https://api.deepinfra.com/v1/openai \
  --api_key YOUR_API_KEY \
  --model allenai/olmOCR-2-7B-1025 \
  --pdfs /path/to/your/*.pdf
```

## ?? Cost

~$0.0003 per page = $0.30 per 1000 pages

**This is cheaper than running your T4 VM!**

## Other Providers

**Cirrascale** (cheapest at $0.07/M tokens):
```bash
python -m olmocr.pipeline ./workspace --markdown \
  --server https://ai2endpoints.cirrascale.ai/api \
  --api_key YOUR_API_KEY \
  --model olmOCR-2-7B-1025 \
  --pdfs sample.pdf
```

**Parasail** ($0.10/M tokens):
```bash
python -m olmocr.pipeline ./workspace --markdown \
  --server https://api.parasail.io/v1 \
  --api_key YOUR_API_KEY \
  --model allenai/olmOCR-2-7B-1025 \
  --pdfs sample.pdf
```

## FAQ

**Q: Can I run locally on T4?**  
A: No, not with the current FP8 models. You'd need an A100, H100, RTX 4090, or similar.

**Q: What about memory optimization?**  
A: Doesn't help. This is a hardware limitation, not a memory issue.

**Q: Is cloud inference secure?**  
A: Yes, but check each provider's privacy policy if you have sensitive documents.

**Q: Is it fast?**  
A: Yes, similar speed to local. ~10-30 seconds per page.

**Q: Do I need to keep my VM running?**  
A: No! You can run olmocr from your local laptop if you want. No GPU needed for cloud inference.

## Need More Help?

See these detailed guides:
- `T4_FP8_ISSUE_FIX.md` - Full explanation of the problem
- `QUICK_COMMANDS.md` - All command options
- `T4_OPTIMIZATION_GUIDE.md` - Original guide (now outdated for T4)

## Bottom Line

**Don't waste time fighting the T4. Use cloud inference. It's easier, cheaper, and works perfectly.**

1. Sign up for DeepInfra (2 min)
2. Copy commands above
3. Replace YOUR_API_KEY
4. Done! ??
