"""Download GLM-OCR model and test it on one page."""

import sys, os, time
import torch
from transformers import AutoProcessor, GlmOcrForConditionalGeneration

model_id = "zai-org/GLM-OCR"

print("Loading processor...")
t0 = time.time()
processor = AutoProcessor.from_pretrained(model_id)
print(f"  done in {time.time() - t0:.1f}s")
print(f"  image_token: {processor.image_token!r}")
print(f"  image_token_id: {processor.image_token_id}")

print("Loading model...")
t0 = time.time()
model = GlmOcrForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
print(f"  done in {time.time() - t0:.1f}s")

# Convert first PDF page to image
import fitz

doc = fitz.open("pdf_idk.pdf")
page = doc[0]
pix = page.get_pixmap(dpi=150)
from PIL import Image

img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
print(f"Image: {img.size}")

# GLM-OCR uses <image> token in text
text = (
    processor.image_token + "\nExtract all text and formulas from this document page."
)
print(f"Prompt: {text[:100]}...")

t0 = time.time()
inputs = processor(images=img, text=text, return_tensors="pt").to(model.device)
print(f"  processed in {time.time() - t0:.1f}s")
print(f"  input shapes: { {k: v.shape for k, v in inputs.items()} }")

t0 = time.time()
generated_ids = model.generate(**inputs, max_new_tokens=2048)
print(f"  inference in {time.time() - t0:.1f}s")

result = processor.decode(generated_ids[0], skip_special_tokens=True)
print(f"\nResult ({len(result)} chars):")
print(result[:1000])
print("...")
doc.close()
