"""Phase 1: Layout segmentation using PP-DocLayoutV3.
Outputs zone metadata JSON for use by downstream phases.

Usage:
  python phase1/segment.py pdf_idk.pdf -o ./phases
"""

import os, json, time, sys, argparse, io
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image


def pdf_to_images(pdf_path: str, dpi: int = 200) -> tuple:
    import fitz

    doc = fitz.open(pdf_path)
    images = []
    pdf_dims = []
    for i in range(len(doc)):
        page = doc[i]
        pdf_dims.append({"width": page.rect.width, "height": page.rect.height})
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    doc.close()
    return images, pdf_dims


def run_layout_detection(page_images, device="cpu", threshold=0.3):
    from glmocr.config import LayoutConfig
    from glmocr.layout import PPDocLayoutDetector

    cfg = LayoutConfig(
        model_dir="PaddlePaddle/PP-DocLayoutV3_safetensors",
        threshold=threshold,
        batch_size=4,
        device=device,
        cuda_visible_devices="0",
        layout_nms=True,
        label_task_mapping={
            "text": [
                "abstract",
                "algorithm",
                "content",
                "doc_title",
                "figure_title",
                "paragraph_title",
                "reference_content",
                "text",
                "vertical_text",
                "vision_footnote",
                "seal",
                "formula_number",
            ],
            "table": ["table"],
            "formula": ["display_formula", "inline_formula"],
            "skip": ["chart", "image"],
            "abandon": [
                "header",
                "footer",
                "number",
                "footnote",
                "aside_text",
                "reference",
                "footer_image",
                "header_image",
            ],
        },
        id2label={
            k: str(v)
            for k, v in {
                0: "abstract",
                1: "algorithm",
                2: "aside_text",
                3: "chart",
                4: "content",
                5: "display_formula",
                6: "doc_title",
                7: "figure_title",
                8: "footer",
                9: "footer_image",
                10: "footnote",
                11: "formula_number",
                12: "header",
                13: "header_image",
                14: "image",
                15: "inline_formula",
                16: "number",
                17: "paragraph_title",
                18: "reference",
                19: "reference_content",
                20: "seal",
                21: "table",
                22: "text",
                23: "vertical_text",
                24: "vision_footnote",
            }.items()
        },
    )
    detector = PPDocLayoutDetector(cfg)
    detector.start()
    try:
        results, vis_images = detector.process(page_images, save_visualization=True)
        return results, vis_images
    finally:
        detector.stop()


def cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _read_config(key: str, default: str = "") -> str:
    config_path = Path(__file__).resolve().parent.parent / "config.txt"
    if config_path.exists():
        for line in config_path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith(key):
                return line.split("=", 1)[1].strip()
    return default


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Layout Segmentation")
    parser.add_argument("pdf", help="Path to PDF file")
    parser.add_argument(
        "--output-dir", "-o", default="./phases", help="Output directory"
    )
    parser.add_argument("--device", default="cpu", help="Device for PP-DocLayoutV3")
    parser.add_argument(
        "--threshold", type=float, default=0.3, help="Detection threshold"
    )
    parser.add_argument("--dpi", type=int, default=200, help="PDF rendering DPI")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save visualization PDF with colored zones",
    )
    parser.add_argument(
        "--lm-studio",
        nargs="?",
        const="auto",
        default=None,
        metavar="URL",
        help="Use LM Studio vision model for layout detection (optionally specify URL)",
    )
    parser.add_argument(
        "--lm-model",
        default="",
        help="LM Studio model name for layout detection"
    )
    args = parser.parse_args()

    if not os.path.isfile(args.pdf):
        print(f"Error: PDF not found: {args.pdf}")
        sys.exit(1)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pdf_stem = Path(args.pdf).stem
    print(f"[Phase 1] Segmenting {args.pdf}...")
    t0 = time.time()

    page_images, pdf_dims = pdf_to_images(args.pdf, dpi=args.dpi)
    print(f"  {len(page_images)} pages loaded in {time.time() - t0:.1f}s")

    # Determine layout engine
    lm_url = None
    if args.lm_studio == "auto":
        # Auto-detect: use LM Studio only when CUDA is unavailable
        config_lm = _read_config("LM_STUDIO_HOST", "")
        if not cuda_available() and config_lm:
            lm_url = config_lm
            print(f"  CUDA not detected, falling back to LM Studio layout: {lm_url}")
    elif args.lm_studio:
        lm_url = args.lm_studio
    elif args.lm_studio is None:
        # Not specified; check config.txt for LM_STUDIO_PHASE1
        config_val = _read_config("LM_STUDIO_PHASE1_HOST", "")
        if config_val:
            lm_url = config_val
            print(f"  Using LM Studio layout (config): {config_val}")

    t1 = time.time()
    if lm_url:
        from phase1.lm_layout import run_lm_layout_detection

        model = args.lm_model or _read_config("LM_STUDIO_PHASE1_MODEL", "") or "mistralai/ministral-3-3b"
        print(f"  Layout engine: LM Studio ({lm_url})")
        print(f"  Model: {model}")
        layout_results, vis_images = run_lm_layout_detection(
            page_images, lm_url=lm_url, model=model, dpi=args.dpi,
        )
    else:
        device = args.device
        if device == "cuda" and not cuda_available():
            device = "cpu"
            print(f"  CUDA requested but not available. Falling back to CPU.")
        print(f"  Layout engine: PP-DocLayoutV3 ({device})")
        layout_results, vis_images = run_layout_detection(
            page_images, device=device, threshold=args.threshold
        )

    total = sum(len(r) for r in layout_results)
    print(f"  {total} regions detected in {time.time() - t1:.1f}s")

    # Filter out nested zones (strict containment)
    def is_nested(small, large):
        ax1, ay1, ax2, ay2 = small
        bx1, by1, bx2, by2 = large
        return ax1 >= bx1 and ay1 >= by1 and ax2 <= bx2 and ay2 <= by2

    def filter_nested(regions):
        kept = []
        areas = []
        for r in regions:
            b = r["bbox_2d"]
            areas.append((b[2] - b[0]) * (b[3] - b[1]))
        for i, r in enumerate(regions):
            b = r["bbox_2d"]
            nested = False
            for j, other in enumerate(regions):
                if i == j:
                    continue
                ob = other["bbox_2d"]
                if areas[i] <= areas[j] and is_nested(b, ob):
                    nested = True
                    break
            if not nested:
                kept.append(r)
        return kept

    filtered_results = [filter_nested(page) for page in layout_results]
    total_kept = sum(len(r) for r in filtered_results)
    removed = total - total_kept
    print(f"  Removed {removed} nested zones, {total_kept} remaining")

    # Regenerate visualization with filtered zones
    import copy
    import numpy as np

    def draw_filtered_viz(img, regions, id2label):
        import cv2

        vis = np.array(img.convert("RGB"))
        colors = {
            "text": (0, 255, 0),
            "table": (255, 0, 0),
            "formula": (0, 0, 255),
            "skip": (255, 255, 0),
        }
        h, w = vis.shape[:2]
        for r in regions:
            label = r["label"]
            task = r["task_type"]
            color = colors.get(task, (128, 128, 128))
            x1 = int(r["bbox_2d"][0] * w / 1000)
            y1 = int(r["bbox_2d"][1] * h / 1000)
            x2 = int(r["bbox_2d"][2] * w / 1000)
            y2 = int(r["bbox_2d"][3] * h / 1000)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                vis,
                label,
                (x1, max(y1 - 2, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
            )
        return Image.fromarray(vis)

    id2label = {
        0: "abstract",
        1: "algorithm",
        2: "aside_text",
        3: "chart",
        4: "content",
        5: "display_formula",
        6: "doc_title",
        7: "figure_title",
        8: "footer",
        9: "footer_image",
        10: "footnote",
        11: "formula_number",
        12: "header",
        13: "header_image",
        14: "image",
        15: "inline_formula",
        16: "number",
        17: "paragraph_title",
        18: "reference",
        19: "reference_content",
        20: "seal",
        21: "table",
        22: "text",
        23: "vertical_text",
        24: "vision_footnote",
    }

    filtered_vis = {}
    for page_idx, regions in enumerate(filtered_results):
        filtered_vis[page_idx] = draw_filtered_viz(
            page_images[page_idx], regions, id2label
        )

    import fitz

    viz_doc = fitz.open()
    for page_idx in sorted(filtered_vis.keys()):
        vis_img = filtered_vis[page_idx]
        pw = pdf_dims[page_idx]["width"]
        ph = pdf_dims[page_idx]["height"]
        img_bytes = io.BytesIO()
        vis_img.save(img_bytes, format="PNG")
        viz_page = viz_doc.new_page(width=pw, height=ph)
        viz_page.insert_image(viz_page.rect, stream=img_bytes.getvalue())
    viz_path = out / "phase1_viz.pdf"
    viz_doc.save(str(viz_path), deflate=True)
    viz_doc.close()
    print(f"  Saved filtered visualization to {viz_path}")

    layout_results = filtered_results

    zones = []
    for page_idx, regions in enumerate(layout_results):
        sorted_regions = sorted(
            regions, key=lambda r: (r["bbox_2d"][1], r["bbox_2d"][0])
        )
        for region in sorted_regions:
            zones.append(
                {
                    "page": page_idx,
                    "label": region["label"],
                    "task_type": region["task_type"],
                    "bbox_norm": region["bbox_2d"],
                }
            )

    payload = {
        "pdf_stem": pdf_stem,
        "pdf_path": str(Path(args.pdf).resolve()),
        "num_pages": len(page_images),
        "pdf_dims": pdf_dims,
        "zones": zones,
    }

    zone_path = out / "phase1_zones.json"
    zone_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  Saved {len(zones)} zones to {zone_path}")
    print(f"[Phase 1] Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
