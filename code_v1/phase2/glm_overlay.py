"""Phase 2: GLM-OCR text/formula/table recognition + PDF text overlay.
Reads zones from Phase 1, runs GLM-OCR, overlays selectable text into PDF.

Usage:
  python phase2/glm_overlay.py phases/phase1_zones.json -o ./phases
  python phase2/glm_overlay.py phases/phase1_zones.json -o ./phases --invisible
"""

import os, json, time, sys, argparse, io
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image, ImageDraw, ImageFont
import fitz
from gpu_utils import free_gpu_memory


def load_glm_ocr():
    from transformers import AutoProcessor, GlmOcrForConditionalGeneration

    model_id = "zai-org/GLM-OCR"
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = GlmOcrForConditionalGeneration.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    model.eval()
    print(f"  GLM-OCR loaded in {time.time() - t0:.1f}s on {model.device}")
    return processor, model


TASK_PROMPTS = {
    "text": "Text Recognition:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
}


def recognize_region(crop_img, task_type, processor, model, max_new_tokens=1024):
    prompt_text = TASK_PROMPTS.get(task_type, "Text Recognition:")
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt_text}],
        }
    ]
    prompt = processor.tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    inputs = processor(images=[crop_img], text=prompt, return_tensors="pt").to(
        model.device
    )
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
    gen_tokens = generated_ids[0][input_len:]
    return processor.decode(gen_tokens, skip_special_tokens=True).strip()


def crop_region(page_img, bbox_norm):
    w, h = page_img.size
    x1 = int(bbox_norm[0] * w / 1000)
    y1 = int(bbox_norm[1] * h / 1000)
    x2 = int(bbox_norm[2] * w / 1000)
    y2 = int(bbox_norm[3] * h / 1000)
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return page_img.crop((0, 0, 10, 10))
    return page_img.crop((x1, y1, x2, y2))


def pdf_page_to_image(pdf_doc, page_idx, dpi=200):
    page = pdf_doc[page_idx]
    pix = page.get_pixmap(dpi=dpi)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def _calc_fontsize_px(text, zone_w_px, zone_h_px):
    """Calculate font size in points for PIL rendering (assumes 72 DPI).

    PIL's textlength returns pixel width at 72 DPI. zone_w_px is the
    zone width on the image (typically 200 DPI). The returned value
    works for PIL draw.text(). For PDF insert_textbox, convert with
    fontsize_pt = fontsize * 72 / dpi.
    """
    if not text:
        return 8
    try:
        ref_font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 100)
    except Exception:
        ref_font = ImageFont.load_default()
    ref_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    w_at_100 = ref_draw.textlength(text, font=ref_font)
    if w_at_100 <= 0:
        w_at_100 = len(text) * 55
    fontsize = zone_w_px / (w_at_100 / 100)
    fontsize = min(fontsize, zone_h_px / 1.2)
    fontsize = max(4, min(200, fontsize))
    return fontsize


def overlay_text_on_image(pil_img, zone, dpi=200):
    """Draw text on PIL image at pixel coordinates (same as phase1_viz.pdf)."""
    bbox = zone["bbox_norm"]
    pw, ph = pil_img.size
    x1 = int(bbox[0] * pw / 1000)
    y1 = int(bbox[1] * ph / 1000)
    x2 = int(bbox[2] * pw / 1000)
    y2 = int(bbox[3] * ph / 1000)

    text = zone.get("content", "").strip()
    if not text:
        return
    text = text.replace("\n", " ")

    zone_w = x2 - x1
    zone_h = y2 - y1
    if zone_w <= 0 or zone_h <= 0:
        return

    fontsize_px = _calc_fontsize_px(text, zone_w, zone_h)
    fontsize_px = int(fontsize_px)

    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", fontsize_px)
    except Exception:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(pil_img)
    if text.strip():
        line_h = fontsize_px * 1.2
        y_center = y1 + (zone_h - line_h) / 2
        draw.text((x1, y_center), text, fill=(0, 0, 0), font=font)


def overlay_text_invisible(page, zone, pdf_dims, pil_img, dpi=200):
    """Add invisible but selectable PDF text. Uses same pixel coords as visible overlay."""
    bbox = zone["bbox_norm"]
    pw = pdf_dims["width"]
    ph = pdf_dims["height"]
    img_w, img_h = pil_img.size

    px1 = int(bbox[0] * img_w / 1000)
    py1 = int(bbox[1] * img_h / 1000)
    px2 = int(bbox[2] * img_w / 1000)
    py2 = int(bbox[3] * img_h / 1000)

    pdf_x1 = px1 * pw / img_w
    pdf_x2 = px2 * pw / img_w
    pdf_y1 = (img_h - py2) * ph / img_h
    pdf_y2 = (img_h - py1) * ph / img_h

    text = zone.get("content", "").strip()
    if not text:
        return
    text = text.replace("\n", " ")

    zone_w_px = px2 - px1
    zone_h_px = py2 - py1
    if zone_w_px <= 0 or zone_h_px <= 0:
        return

    fontsize_px = _calc_fontsize_px(text, zone_w_px, zone_h_px)
    fontsize_pt = fontsize_px * 72 / dpi

    rect = fitz.Rect(pdf_x1, pdf_y1, pdf_x2, pdf_y2)
    result = page.insert_textbox(
        rect, text, fontsize=fontsize_pt, fontname="helv", render_mode=3
    )
    if result < 0:
        print(
            f"    overflow in '{zone['label']}' (ret={result:.1f}) at fontsize={fontsize_pt:.1f}pt, retrying..."
        )
        for retry in range(8):
            fontsize_pt *= 0.65
            result = page.insert_textbox(
                rect, text, fontsize=fontsize_pt, fontname="helv", render_mode=3
            )
            if result >= 0:
                break
        if result < 0:
            print(
                f"    WARNING: still overflow after retries for '{zone['label']}' (ret={result:.1f})"
            )


def main():
    parser = argparse.ArgumentParser(description="Phase 2: GLM-OCR + PDF Overlay")
    parser.add_argument("zones_json", help="Phase 1 zones JSON file")
    parser.add_argument(
        "--output-dir", "-o", default="./phases", help="Output directory"
    )
    parser.add_argument(
        "--dpi", type=int, default=200, help="PDF rendering DPI for cropping"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=1024, help="GLM-OCR max tokens"
    )
    parser.add_argument(
        "--invisible",
        action="store_true",
        help="Make overlay text invisible (render_mode=3)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.zones_json):
        print(f"Error: zones JSON not found: {args.zones_json}")
        sys.exit(1)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(args.zones_json, encoding="utf-8") as f:
        data = json.load(f)

    pdf_path = data["pdf_path"]
    pdf_dims = data["pdf_dims"]
    zones = data["zones"]

    if not os.path.isfile(pdf_path):
        print(f"Error: PDF not found at recorded path: {pdf_path}")
        sys.exit(1)

    print(f"[Phase 2] GLM-OCR recognition + PDF overlay")
    print(f"  PDF: {pdf_path}")
    print(f"  Zones: {len(zones)}")
    print(f"  Text overlay: {'invisible' if args.invisible else 'visible'}")
    t0 = time.time()

    free_gpu_memory(verbose=True)
    src_doc = fitz.open(pdf_path)
    page_images = []
    for page_idx in range(len(src_doc)):
        src_page = src_doc[page_idx]
        pix = src_page.get_pixmap(dpi=args.dpi)
        page_images.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
    src_doc.close()

    processor, model = load_glm_ocr()
    ocr_zones = 0
    try:
        for zi, zone in enumerate(zones):
            task_type = zone["task_type"]
            if task_type == "skip":
                zone["content"] = ""
                zone["source"] = "pending"
                continue

            page_idx = zone["page"]
            if page_idx >= len(page_images):
                continue

            page_img = page_images[page_idx]
            crop_img = crop_region(page_img, zone["bbox_norm"])
            text = recognize_region(
                crop_img, task_type, processor, model, max_new_tokens=args.max_tokens
            )
            zone["content"] = text
            zone["source"] = "glm_ocr"
            ocr_zones += 1

            preview = text[:80].replace("\n", " ")
            if len(preview) > 80:
                preview = preview[:77] + "..."
            print(f"  Page {page_idx + 1} [{zone['label']}]: {preview}")

    finally:
        del processor, model
        free_gpu_memory(verbose=True)

    clean_doc = fitz.open()
    for page_idx, pil_img in enumerate(page_images):
        pw = pdf_dims[page_idx]["width"]
        ph = pdf_dims[page_idx]["height"]

        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        new_page = clean_doc.new_page(width=pw, height=ph)
        new_page.insert_image(new_page.rect, stream=buf.getvalue())

        if args.invisible:
            for z in zones:
                if z["page"] != page_idx:
                    continue
                if not z.get("content", "").strip():
                    continue
                overlay_text_invisible(
                    new_page, z, pdf_dims[page_idx], pil_img, dpi=args.dpi
                )
        else:
            for z in zones:
                if z["page"] != page_idx:
                    continue
                if not z.get("content", "").strip():
                    continue
                overlay_text_on_image(pil_img, z, dpi=args.dpi)
                overlay_text_invisible(
                    new_page, z, pdf_dims[page_idx], pil_img, dpi=args.dpi
                )

    ocr_pdf = out / "phase2_ocrd.pdf"
    clean_doc.save(str(ocr_pdf), deflate=True)
    clean_doc.close()
    print(f"  Saved OCR'd PDF ({ocr_pdf})")

    data["zones"] = zones
    zone_out = out / "phase2_zones.json"
    zone_out.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  {ocr_zones} zones recognized in {time.time() - t0:.1f}s")
    print(f"[Phase 2] Done")


if __name__ == "__main__":
    main()
