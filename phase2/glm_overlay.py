"""Phase 2: GLM-OCR text/formula/table recognition + PDF text overlay.
Reads zones from Phase 1, runs GLM-OCR, overlays selectable text into PDF.

Usage:
  python phase2/glm_overlay.py phases/phase1_zones.json -o ./phases
  python phase2/glm_overlay.py phases/phase1_zones.json -o ./phases --invisible
"""

import os, json, time, sys, argparse, io, re
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image, ImageDraw, ImageFont
import fitz
from gpu_utils import free_gpu_memory


def load_glm_ocr():
    import os
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_SYSTEM_PROMPTS"] = "1"
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


def normalize_ocr_text(text, task_type):
    """Clean OCR output while preserving line structure."""
    if not text:
        return ""
    text = text.replace("\r", "\n")
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r"\n\s*\n", "\n", text)
    text = text.strip()
    if task_type == "formula":
        text = text.replace("$$ ", "$$").replace(" $$", "$$").replace("\n", " ")
    return text


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


def _draw_centered_text(draw, pil_img, zone, text, font, fill=(0, 0, 0)):
    bbox = zone["bbox_norm"]
    pw, ph = pil_img.size
    x1 = int(bbox[0] * pw / 1000)
    y1 = int(bbox[1] * ph / 1000)
    x2 = int(bbox[2] * pw / 1000)
    y2 = int(bbox[3] * ph / 1000)
    zone_w = x2 - x1
    zone_h = y2 - y1
    if zone_w <= 0 or zone_h <= 0:
        return

    try:
        tb = draw.textbbox((0, 0), text, font=font)
        text_w = tb[2] - tb[0]
        text_h = tb[3] - tb[1]
    except Exception:
        text_w = int(len(text) * 10)
        text_h = 12

    x = x1 + max(0, (zone_w - text_w) / 2)
    y = y1 + max(0, (zone_h - text_h) / 2)
    draw.text((x, y), text, fill=fill, font=font)


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
        _draw_centered_text(draw, pil_img, zone, text, font)


def overlay_text_selectable(page, zone, pdf_dims, pil_img, dpi=200):
    """Add invisible-but-selectable PDF text centered in zone."""
    bbox_norm = zone["bbox_norm"]
    pw = pdf_dims["width"]
    ph = pdf_dims["height"]

    text = zone.get("content", "").strip()
    if not text:
        return
    text = normalize_ocr_text(text, zone.get("task_type", "text"))

    fitz_font = fitz.Font("helv")
    asc = fitz_font.ascender
    desc = fitz_font.descender
    zone_w = (bbox_norm[2] - bbox_norm[0]) * pw / 1000
    zone_h = (bbox_norm[3] - bbox_norm[1]) * ph / 1000
    if zone_w <= 0 or zone_h <= 0:
        return

    tl = fitz_font.text_length(text, fontsize=1)
    if tl <= 0:
        return

    fontsize = zone_w / tl

    text_w = fitz_font.text_length(text, fontsize=fontsize)
    x = bbox_norm[0] * pw / 1000 + max(0, (zone_w - text_w) / 2)

    y_center = (bbox_norm[1] + bbox_norm[3]) / 2 * ph / 1000
    y = y_center + (asc + desc) / 2 * fontsize

    page.insert_text(
        (x, y),
        text,
        fontsize=fontsize,
        fontname="helv",
        render_mode=3,
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
    parser.add_argument(
        "--skip-ocr",
        action="store_true",
        help="Skip GLM-OCR, reuse existing zone content from JSON",
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

    ocr_zones = 0
    if not args.skip_ocr:
        processor, model = load_glm_ocr()
        try:
            ocr_total = sum(1 for z in zones if z.get("task_type") != "skip")
            ocr_done = 0
            expanded_zones = []
            for zi, zone in enumerate(zones):
                task_type = zone["task_type"]
                if task_type == "skip":
                    zone["content"] = ""
                    zone["source"] = "pending"
                    expanded_zones.append(zone)
                    continue

                page_idx = zone["page"]
                if page_idx >= len(page_images):
                    expanded_zones.append(zone)
                    continue

                page_img = page_images[page_idx]
                crop_img = crop_region(page_img, zone["bbox_norm"])
                text = recognize_region(
                    crop_img, task_type, processor, model,
                    max_new_tokens=args.max_tokens,
                )
                zone["content"] = normalize_ocr_text(text, task_type)
                zone["source"] = "glm_ocr"
                ocr_zones += 1
                ocr_done += 1

                lines = [l for l in zone["content"].split("\n") if l.strip()]
                if len(lines) > 1 and task_type != "formula":
                    bbox = zone["bbox_norm"]
                    y1, y2 = bbox[1], bbox[3]
                    line_h = (y2 - y1) / len(lines)
                    for li, line_text in enumerate(lines):
                        sub = dict(zone)
                        sub["content"] = line_text.strip()
                        sub["bbox_norm"] = [
                            bbox[0], y1 + li * line_h,
                            bbox[2], y1 + (li + 1) * line_h,
                        ]
                        sub["label"] = f"{zone['label']}_L{li+1}"
                        expanded_zones.append(sub)
                    preview = zone["content"][:80].replace("\n", " | ")
                    print(f"  [{ocr_done}/{ocr_total}] Page {page_idx + 1} [{zone['label']}]: {preview}  [{len(lines)} lines split]")
                else:
                    expanded_zones.append(zone)
                    preview = zone["content"][:80].replace("\n", " ")
                    print(f"  [{ocr_done}/{ocr_total}] Page {page_idx + 1} [{zone['label']}]: {preview}")

            zones = expanded_zones

        finally:
            del processor, model
            free_gpu_memory(verbose=True)
    else:
        ocr_zones = sum(1 for z in zones if z.get("content", "").strip())
        print(f"  Skipping OCR, reusing {ocr_zones} existing zone contents")

    clean_doc = fitz.open()
    viz_path = out / "phase1_viz.pdf"
    if viz_path.is_file():
        viz_doc = fitz.open(str(viz_path))
        print(f"  Using {viz_path} as background")
    else:
        viz_doc = None
        print(f"  No {viz_path} found, rendering from source PDF")

    for page_idx in range(len(page_images)):
        pw = pdf_dims[page_idx]["width"]
        ph = pdf_dims[page_idx]["height"]

        new_page = clean_doc.new_page(width=pw, height=ph)
        if viz_doc and page_idx < len(viz_doc):
            new_page.show_pdf_page(new_page.rect, viz_doc, page_idx)
        else:
            buf = io.BytesIO()
            page_images[page_idx].save(buf, format="PNG")
            new_page.insert_image(new_page.rect, stream=buf.getvalue())

        for z in zones:
            if z["page"] != page_idx:
                continue
            if not z.get("content", "").strip():
                continue
            overlay_text_selectable(
                new_page, z, pdf_dims[page_idx], page_images[page_idx], dpi=args.dpi
            )

    if viz_doc:
        viz_doc.close()

    ocr_pdf = out / "phase2_ocrd_v2.pdf"
    clean_doc.save(str(ocr_pdf), deflate=True)
    clean_doc.close()
    print(f"  Saved OCR'd PDF ({ocr_pdf})")

    data["zones"] = zones
    zone_out = out / "phase2_zones_v2.json"
    zone_out.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  {ocr_zones} zones recognized in {time.time() - t0:.1f}s")
    print(f"[Phase 2] Done")


if __name__ == "__main__":
    main()
