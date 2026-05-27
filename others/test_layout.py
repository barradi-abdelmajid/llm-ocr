"""Quick test: layout detection on first 2 pages."""

import sys, time

sys.path.insert(0, ".")
from glmocr_pipeline import pdf_to_images, run_layout_detection

pages, pdf_dims = pdf_to_images("pdf_idk.pdf", dpi=200)
print(f"Loaded {len(pages)} pages")

results = run_layout_detection(pages[:2], device="cpu")
for pi, regions in enumerate(results):
    print(f"Page {pi + 1}: {len(regions)} regions")
    for ri, r in enumerate(regions):
        print(f"  [{ri}] label={r['label']} task={r['task_type']} bbox={r['bbox_2d']}")
