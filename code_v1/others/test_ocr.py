"""Test GLM-OCR recognition on a few regions from page 2."""

import sys, time

sys.path.insert(0, ".")
from glmocr_pipeline import (
    pdf_to_images,
    run_layout_detection,
    crop_region,
    load_glm_ocr,
    recognize_region,
)

pages, pdf_dims = pdf_to_images("pdf_idk.pdf", dpi=200)
print(f"Loaded {len(pages)} pages")

results = run_layout_detection(pages[:2], device="cpu")

# Test OCR on page 2, text regions
page_idx = 1
text_regions = [r for r in results[page_idx] if r["task_type"] != "skip"]

print(f"\nPage 2 has {len(text_regions)} non-skip regions")

processor, model = load_glm_ocr()

for i, region in enumerate(text_regions[:3]):
    crop = crop_region(pages[page_idx], region["bbox_2d"])
    t0 = time.time()
    text = recognize_region(crop, region["task_type"], processor, model)
    elapsed = time.time() - t0
    print(f"\n[{i}] label={region['label']} bbox={region['bbox_2d']} ({elapsed:.1f}s)")
    print(f"    text: {text[:200]}")

del processor, model
if __name__ != "__main__":
    import torch

    torch.cuda.empty_cache()
