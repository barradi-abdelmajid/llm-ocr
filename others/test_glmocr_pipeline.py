"""Test GLM-OCR pipeline for layout detection."""

import os, sys, json, time

# Patch: force numpy < 2.4 for paddlex compat
import numpy as np

if np.__version__ >= "2.4":
    pass  # will try anyway

from glmocr.config import load_config, GlmOcrConfig
from glmocr.pipeline.pipeline import Pipeline

config = load_config()

# Override: use cloud API (needs env ZHIPU_API_KEY or pass --api-key)
config["pipeline"]["maas"]["enabled"] = False
config["pipeline"]["ocr_api"]["api_host"] = "localhost"
config["pipeline"]["ocr_api"]["api_port"] = 8000
config["pipeline"]["ocr_api"]["api_key"] = "not-needed"

cfg = GlmOcrConfig(**config)

print("Initializing pipeline...")
t0 = time.time()
pipe = Pipeline(config=cfg.pipeline)
print(f"  done in {time.time() - t0:.1f}s")

from glmocr.dataloader import PageLoader

# Load a PDF
pdf_path = "pdf_idk.pdf"
page_loader = PageLoader(cfg.pipeline.page_loader)
pages = page_loader.load(pdf_path)
print(f"Loaded {len(pages)} pages")

# Process first page
page = pages[0]
print(f"Page 1: {page.width}x{page.height}")

# Run layout detection
t0 = time.time()
layout_result = pipe.layout_detector.detect(page)
print(f"Layout detection: {time.time() - t0:.1f}s")
print(f"Detected {len(layout_result)} regions")

for i, region in enumerate(layout_result):
    print(f"  {i}: label={region.label} bbox={region.bbox}")

# Save a debug visualization
from glmocr.utils.visualize import draw_layout_result

vis = draw_layout_result(page, layout_result)
vis.save("debug_glm_layout.png")
print("Saved debug_glm_layout.png")
