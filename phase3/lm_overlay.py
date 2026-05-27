"""Phase 3: LM Studio image/chart recognition + PDF text overlay.
Reads zones + partially OCR'd PDF from Phase 2, sends image/chart zones to LM Studio,
inserts descriptions into PDF as selectable text.

Usage:
  python phase3/lm_overlay.py phases/phase2_zones.json -o ./phases
  python phase3/lm_overlay.py phases/phase2_zones.json -o ./phases --invisible
"""

import os, json, time, sys, argparse, io, base64, re, gc
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import fitz
from PIL import Image
from gpu_utils import free_gpu_memory, unload_model


def load_prompts(prompts_path: str) -> dict:
    default_prompts = {
        "image": (
            "Analyze this image. If it's a circuit diagram, output ONE line "
            "in Chinese (Mandarin) starting with '> ' describing it concisely. "
            "Include component values if visible. "
            "Example: '> 串联R=10Ω和V=5V的简单直流电路' "
            "If it's not a circuit, describe briefly in Chinese with '> ' prefix."
        ),
        "image_modification": (
            "This modifies the previous circuit. Output ONE line in Chinese "
            "starting with '> ' describing ONLY what changed. "
            "Example: '> R1替换为2A电流源'"
        ),
        "chart": (
            "Describe this chart in detail, including axes, data trends, "
            "and values in Chinese with '> ' prefix."
        ),
    }

    if not prompts_path or not os.path.isfile(prompts_path):
        return default_prompts

    prompts = {}
    current_section = None
    current_lines = []

    for line in Path(prompts_path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            continue
        if stripped.startswith("# "):
            if current_section and current_lines:
                prompts[current_section] = "\n".join(current_lines).strip()
            current_section = stripped[2:].strip()
            current_lines = []
        elif current_section is not None:
            current_lines.append(line)

    if current_section and current_lines:
        prompts[current_section] = "\n".join(current_lines).strip()

    return prompts if prompts else default_prompts


def group_circuit_regions(zones):
    PROXIMITY = 200
    img_indices = [i for i, z in enumerate(zones) if z["label"] in ("image", "chart")]
    groups = []
    current = []
    for idx in img_indices:
        z = zones[idx]
        if not current:
            current.append(idx)
        else:
            prev_z = zones[current[-1]]
            same_page = z["page"] == prev_z["page"]
            prev_cy = (prev_z["bbox_norm"][1] + prev_z["bbox_norm"][3]) / 2
            curr_cy = (z["bbox_norm"][1] + z["bbox_norm"][3]) / 2
            if same_page and abs(curr_cy - prev_cy) < PROXIMITY:
                current.append(idx)
            else:
                groups.append(current)
                current = [idx]
    if current:
        groups.append(current)
    return groups


def lm_studio_analyze(
    crop_img,
    prompt,
    lm_studio_url,
    model_name="mistralai/ministral-3-3b",
    max_tokens=128,
    timeout=60,
):
    import requests

    buf = io.BytesIO()
    crop_img.save(buf, format="JPEG", quality=95)
    b64 = base64.b64encode(buf.getvalue()).decode()

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }

    try:
        resp = requests.post(
            f"{lm_studio_url.rstrip('/')}/chat/completions",
            json=payload,
            timeout=timeout,
        )
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"].strip()
        return f"[LM Studio error: HTTP {resp.status_code}]"
    except requests.exceptions.ConnectionError:
        return "[LM Studio unavailable]"
    except Exception as e:
        return f"[LM Studio error: {e}]"


def crop_region_from_pdf(pdf_doc, page_idx, bbox_norm, dpi=200):
    page = pdf_doc[page_idx]
    pix = page.get_pixmap(dpi=dpi)
    page_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
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


EST_CHAR_WIDTH = 0.55


def word_wrap(text, max_chars):
    lines = []
    for para in text.split("\n"):
        words = para.split()
        current = ""
        for w in words:
            if len(current) + len(w) + 1 <= max_chars:
                current = (current + " " + w).strip()
            else:
                if current:
                    lines.append(current)
                current = w
        if current:
            lines.append(current)
    return lines if lines else [""]


def overlay_text_on_page(page, zone, pdf_dims, invisible=False):
    bbox = zone["bbox_norm"]
    pw = pdf_dims["width"]
    ph = pdf_dims["height"]

    x1 = bbox[0] * pw / 1000
    x2 = bbox[2] * pw / 1000
    y1 = (1000 - bbox[3]) * ph / 1000
    y2 = (1000 - bbox[1]) * ph / 1000

    text = zone.get("content", "").strip()
    if not text:
        return

    zone_w = x2 - x1
    zone_h = y2 - y1
    if zone_w <= 0 or zone_h <= 0:
        return

    fontname = "helv" if all(ord(c) < 128 for c in text) else "china-s"

    max_chars_per_line = max(1, int(zone_w / (EST_CHAR_WIDTH * 1)))
    wrapped_lines = word_wrap(text, max_chars_per_line)
    num_lines = len(wrapped_lines)

    fontsize = min(
        zone_h / (num_lines + 0.3),
        zone_w / (max_chars_per_line * EST_CHAR_WIDTH + 1),
    )
    fontsize = max(4, min(72, fontsize))

    line_height = fontsize * 1.2

    th = num_lines * line_height
    if th > zone_h:
        scale = zone_h / th
        fontsize *= scale
        line_height = fontsize * 1.2

    if invisible:
        rect = fitz.Rect(x1, y1, x2, y2)
        page.insert_textbox(
            rect, text, fontsize=fontsize, fontname=fontname, render_mode=3
        )
    else:
        y_cursor = y2 - fontsize * 0.2
        for line in wrapped_lines:
            if line.strip():
                page.insert_text(
                    fitz.Point(x1, y_cursor), line, fontsize=fontsize, fontname=fontname
                )
            y_cursor -= line_height


def main():
    parser = argparse.ArgumentParser(description="Phase 3: LM Studio Image Overlay")
    parser.add_argument("zones_json", help="Phase 2 zones JSON file")
    parser.add_argument(
        "--output-dir", "-o", default="./phases", help="Output directory"
    )
    parser.add_argument("--lm-studio", help="LM Studio API URL (overrides config/JSON)")
    parser.add_argument(
        "--prompts", help="Path to prompts file (overrides config/JSON)"
    )
    parser.add_argument("--dpi", type=int, default=200, help="PDF rendering DPI")
    parser.add_argument(
        "--workers", type=int, default=5, help="Parallel LM Studio workers"
    )
    parser.add_argument("--timeout", type=int, default=120, help="LM Studio timeout")
    parser.add_argument(
        "--max-tokens", type=int, default=128, help="LM Studio max tokens per call"
    )
    parser.add_argument(
        "--image-max-tokens",
        type=int,
        default=100,
        help="Max tokens for image/chart descriptions",
    )
    parser.add_argument(
        "--lm-model",
        default="mistralai/ministral-3-3b",
        help="Model name for LM Studio API",
    )
    parser.add_argument(
        "--local-lm",
        help="Run LM locally via transformers (model ID, e.g. mistralai/ministral-3-3b)",
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

    original_pdf = data["pdf_path"]
    if not os.path.isfile(original_pdf):
        print(f"Error: Original PDF not found: {original_pdf}")
        sys.exit(1)

    phase2_pdf = str(out / "phase2_ocrd.pdf")
    if os.path.isfile(phase2_pdf):
        base_pdf = phase2_pdf
        print(f"  Base PDF: Phase 2 OCR'd PDF")
    else:
        base_pdf = original_pdf
        print(f"  Base PDF: Original (no Phase 2 PDF found)")

    pdf_dims = data.get("pdf_dims", data.get("pdf_dims"))
    zones = data["zones"]
    num_pages = data["num_pages"]

    lm_url = args.lm_studio
    if not lm_url:
        config_path = Path("config.txt")
        if config_path.exists():
            for line in config_path.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("LM_STUDIO_HOST"):
                    lm_url = line.split("=", 1)[1].strip()
    if not lm_url:
        print(
            "Error: No LM Studio URL provided. Use --lm-studio or set LM_STUDIO_HOST in config.txt"
        )
        sys.exit(1)

    prompts_path = args.prompts
    if not prompts_path:
        config_path = Path("config.txt")
        if config_path.exists():
            for line in config_path.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("LM_PROMPTS_FILE"):
                    prompts_path = line.split("=", 1)[1].strip()
    prompts = load_prompts(prompts_path)

    print(f"[Phase 3] LM Studio image overlay")
    print(f"  Base PDF: {base_pdf}")
    print(f"  Prompts: {list(prompts.keys())}")
    print(f"  Text overlay: {'invisible' if args.invisible else 'visible'}")
    t0 = time.time()

    use_local = bool(args.local_lm)
    if use_local:
        print(f"  Mode: LOCAL transformers (model: {args.local_lm})")
    else:
        print(f"  LM Studio: {lm_url}")
        import requests

        lm_available = False
        try:
            r = requests.get(f"{lm_url.rstrip('/')}/models", timeout=5)
            lm_available = r.status_code == 200
        except Exception:
            pass
        if not lm_available:
            print(f"  LM Studio not reachable at {lm_url}")
            for z in zones:
                if not z.get("content", "").strip():
                    z["content"] = ""
                    z["source"] = "skipped"
            doc = fitz.open(base_pdf)
            doc.save(str(out / "phase3_complete.pdf"), deflate=True)
            doc.close()
            with open(out / "phase3_zones.json", "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"  Skipped LM Studio. Saved unchanged PDF.")
            print(f"[Phase 3] Done in {time.time() - t0:.1f}s")
            return

    doc = fitz.open(base_pdf)
    original_doc = fitz.open(original_pdf)

    circuit_groups = group_circuit_regions(zones)
    prompt_overrides = {}
    for group in circuit_groups:
        for i, idx in enumerate(group):
            prompt_overrides[idx] = "image" if i == 0 else "image_modification"

    lm_tasks = []
    for i, z in enumerate(zones):
        if z["task_type"] == "skip":
            needs_lm = False
            if z["label"] in ("image", "chart"):
                needs_lm = True
            elif not z.get("content", "").strip():
                needs_lm = True

            if needs_lm:
                prompt_type = prompt_overrides.get(i, z["label"])
                if prompt_type not in prompts:
                    prompt_type = "image"
                prompt_text = prompts[prompt_type]
                lm_tasks.append((i, z, prompt_text))

    dpi = args.dpi
    max_tokens = args.max_tokens
    image_max_tokens = args.image_max_tokens
    lm_timeout = args.timeout
    model_name = args.lm_model

    local_lm_model = None
    local_lm_processor = None
    if use_local:
        print(f"  Loading local model {args.local_lm}...")
        free_gpu_memory(verbose=True)
        from gpu_utils import load_hf_model

        local_lm_model, local_lm_processor = load_hf_model(
            args.local_lm, task_type="vision_lm"
        )
        print(f"  Local model loaded on {local_lm_model.device}")

    if not lm_tasks:
        print(f"  No image/chart zones to process")
    else:
        mode = "Local" if use_local else "LM Studio"
        print(f"  {len(lm_tasks)} zones to analyze via {mode}")

        def process_one_remote(task):
            idx, z, prompt_text = task
            crop_img = crop_region_from_pdf(
                original_doc, z["page"], z["bbox_norm"], dpi=dpi
            )
            mtokens = (
                image_max_tokens
                if prompt_text in ("image", "image_modification")
                else max_tokens
            )
            result = lm_studio_analyze(
                crop_img,
                prompt_text,
                lm_url,
                model_name=model_name,
                max_tokens=mtokens,
                timeout=lm_timeout,
            )
            if result.startswith("[LM Studio"):
                print(f"  LM Studio error for zone {idx}: {result}")
                return (idx, z, None)
            z["content"] = result
            z["source"] = "lm_studio"
            return (idx, z, result)

        def process_one_local(task):
            idx, z, prompt_text = task
            crop_img = crop_region_from_pdf(
                original_doc, z["page"], z["bbox_norm"], dpi=dpi
            )
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            try:
                prompt = local_lm_processor.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
                inputs = local_lm_processor(
                    images=[crop_img], text=prompt, return_tensors="pt"
                ).to(local_lm_model.device)
                input_len = inputs["input_ids"].shape[1]
                with torch.no_grad():
                    generated_ids = local_lm_model.generate(
                        **inputs, max_new_tokens=128, do_sample=False
                    )
                gen_tokens = generated_ids[0][input_len:]
                result = local_lm_processor.decode(
                    gen_tokens, skip_special_tokens=True
                ).strip()
            except Exception as e:
                print(f"  Local LM error for zone {idx}: {e}")
                return (idx, z, None)
            z["content"] = result
            z["source"] = "lm_studio"
            return (idx, z, result)

        worker_fn = process_one_local if use_local else process_one_remote
        with ThreadPoolExecutor(max_workers=1 if use_local else args.workers) as pool:
            futures = [pool.submit(worker_fn, t) for t in lm_tasks]
            for f in as_completed(futures):
                idx, z, result = f.result()
                if result:
                    preview = result[:80].replace("\n", " ")
                    preview = preview.encode("cp1252", errors="replace").decode(
                        "cp1252"
                    )
                    print(f"  LM Page {z['page'] + 1} [{z['label']}]: {preview}")

    if local_lm_model is not None:
        unload_model(local_lm_model, local_lm_processor)
        free_gpu_memory(verbose=True)

    overlay_count = 0
    for z in zones:
        if z.get("source") == "lm_studio" and z.get("content", "").strip():
            overlay_text_on_page(
                doc[z["page"]], z, pdf_dims[z["page"]], invisible=args.invisible
            )
            overlay_count += 1
    print(f"  Overlaid {overlay_count} zones into PDF")

    complete_pdf = out / "phase3_complete.pdf"
    doc.save(str(complete_pdf), deflate=True)
    doc.close()
    original_doc.close()

    data["zones"] = zones
    zone_out = out / "phase3_zones.json"
    zone_out.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  Saved {complete_pdf}")
    print(f"[Phase 3] Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
