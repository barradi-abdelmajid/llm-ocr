"""Phase 3: VLM image/chart recognition + PDF text overlay.
Reads zones + partially OCR'd PDF from Phase 2, analyzes image/chart zones
with a local VLM model (default) or LM Studio, inserts descriptions into PDF.

Usage:
  python phase3/lm_overlay.py phases/phase2_zones.json -o ./phases
  python phase3/lm_overlay.py phases/phase2_zones.json -o ./phases --lm-studio http://localhost:1234/v1
"""

import os, json, time, sys, argparse, io, base64, re, gc
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

warnings.filterwarnings("ignore")

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import fitz
from PIL import Image
from gpu_utils import free_gpu_memory, unload_model


def load_prompts(prompts_path: str) -> dict:
    default_prompts = {
        "image": (
            "Si c'est un schéma de circuit, décris-le en une phrase courte en français. "
            "Mentionne tous les composants avec leurs noms et valeurs. "
            "Exemple: 'Circuit avec R1=10Ω, V=5V et un ampoule' "
            "Sinon, donne une phrase rapide décrivant le contenu. "
            "Ignore les lignes simples ou les éléments sans intérêt. "
            "Maximum 150 tokens."
        ),
        "image_modification": (
            "Décris en une phrase courte en français ce qui a changé par rapport au circuit précédent. "
            "Mentionne les composants modifiés avec leurs nouvelles valeurs. "
            "Exemple: 'R1 remplacé par une source de 2A' "
            "Maximum 150 tokens."
        ),
        "chart": (
            "Décris ce graphique en une phrase courte en français. "
            "Inclus les axes, tendances et valeurs si visibles. "
            "Maximum 150 tokens."
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


_page_cache = {}


def pre_render_pages(pdf_doc, dpi=200):
    """Render all pages to images once and cache them."""
    global _page_cache
    _page_cache = {}
    for page_idx in range(len(pdf_doc)):
        page = pdf_doc[page_idx]
        pix = page.get_pixmap(dpi=dpi)
        _page_cache[page_idx] = Image.frombytes(
            "RGB", [pix.width, pix.height], pix.samples
        )


def crop_region_from_pdf(pdf_doc, page_idx, bbox_norm, dpi=200):
    if page_idx in _page_cache:
        page_img = _page_cache[page_idx]
    else:
        page = pdf_doc[page_idx]
        pix = page.get_pixmap(dpi=dpi)
        page_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        _page_cache[page_idx] = page_img
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

    # Normalize line breaks: replace \r\n / \r with \n
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    fontname = "helv" if all(ord(c) < 128 for c in text) else "china-s"

    # insert_textbox handles word-wrapping, font sizing, and \n line breaks
    rect = fitz.Rect(x1, y1, x2, y2)
    render_mode = 3 if invisible else 0

    # Start with a reasonable font size and shrink if needed
    fontsize = 7.0
    for _ in range(8):
        res = page.insert_textbox(
            rect, text, fontsize=fontsize, fontname=fontname,
            render_mode=render_mode, align=fitz.TEXT_ALIGN_LEFT,
        )
        # res < 0 means text overflowed the rect
        if res >= 0:
            break
        fontsize *= 0.85
    else:
        # Last resort: force-fit with tiny font
        page.insert_textbox(
            rect, text, fontsize=4.0, fontname=fontname,
            render_mode=render_mode, align=fitz.TEXT_ALIGN_LEFT,
        )


def main():
    parser = argparse.ArgumentParser(description="Phase 3: VLM image analysis + overlay")
    parser.add_argument("zones_json", help="Phase 2 zones JSON file")
    parser.add_argument("--output-dir", "-o", default="./phases")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--visible", action="store_true",
                        help="Show overlay text (debug only, default: invisible)")
    parser.add_argument("--prompts", help="Custom prompts txt file")
    parser.add_argument("--local-model", default="Qwen/Qwen3-VL-2B-Instruct",
                        help="Local HuggingFace VLM model (default: Qwen3-VL-2B-Instruct)")
    parser.add_argument("--local-lm", default="Qwen/Qwen3-VL-2B-Instruct",
                        help="Alias for --local-model")
    parser.add_argument("--lm-studio", help="LM Studio URL (optional, overrides local model)")
    parser.add_argument("--lm-model", default="mistralai/ministral-3-3b")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--image-max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Enable 4-bit quantization (requires bitsandbytes + CUDA)")
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

    phase2_pdf = str(out / "phase2_ocrd_v2.pdf")
    if not os.path.isfile(phase2_pdf):
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

    prompts_path = args.prompts
    if not prompts_path:
        config_path = Path("config.txt")
        if config_path.exists():
            for line in config_path.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("LM_PROMPTS_FILE"):
                    prompts_path = line.split("=", 1)[1].strip()
    prompts = load_prompts(prompts_path)

    use_local = not lm_url
    t0 = time.time()

    if use_local:
        print(f"[Phase 3] VLM image overlay (local model)")
        print(f"  Model: {args.local_model}")
    else:
        print(f"[Phase 3] VLM image overlay (LM Studio)")
        print(f"  URL: {lm_url}")

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

    print(f"  Base PDF: {base_pdf}")
    print(f"  Prompts: {list(prompts.keys())}")
    print(f"  Text overlay: {'invisible' if args.invisible else 'visible'}")
    t0 = time.time()

    doc = fitz.open(base_pdf)
    original_doc = fitz.open(original_pdf)

    dpi = args.dpi
    print(f"  Pre-rendering {num_pages} pages at {dpi} DPI...")
    pre_render_pages(original_doc, dpi=dpi)

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

    max_tokens = args.max_tokens
    image_max_tokens = args.image_max_tokens
    lm_timeout = args.timeout
    model_name = args.lm_model

    local_lm_model = None
    local_lm_processor = None
    if use_local:
        model_id = args.local_lm or args.local_model
        quant = " (4-bit)" if args.load_in_4bit else ""
        print(f"  Loading local model {model_id}{quant}...")
        free_gpu_memory(verbose=True)
        from gpu_utils import load_hf_model

        local_lm_model, local_lm_processor = load_hf_model(
            model_id, task_type="vision_lm", load_in_4bit=args.load_in_4bit
        )
        print(f"  Local model loaded on {local_lm_model.device}")

    if not lm_tasks:
        print(f"  No image/chart zones to process")
    else:
        mode = "Local" if use_local else "LM Studio"
        print(f"  {len(lm_tasks)} zones to analyze via {mode}")

        _progress_remote = [0]

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
            t_start = time.time()
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
            _progress_remote[0] += 1
            elapsed = time.time() - t_start
            preview = result[:80].replace("\n", " ")
            try:
                preview = preview.encode("cp1252", errors="replace").decode("cp1252")
            except Exception:
                pass
            print(f"  [{_progress_remote[0]}/{len(lm_tasks)}] Page {z['page']+1} [{z['label']}]: {preview} ({elapsed:.1f}s)")
            return (idx, z, result)

        _progress = [0]

        def process_one_local(task):
            idx, z, prompt_text = task
            crop_img = crop_region_from_pdf(
                original_doc, z["page"], z["bbox_norm"], dpi=dpi
            )
            w, h = crop_img.size
            if max(w, h) > 768:
                scale = 768 / max(w, h)
                crop_img = crop_img.resize(
                    (int(w * scale), int(h * scale)), Image.LANCZOS
                )
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": crop_img},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            mtokens = (
                128 if prompt_text in ("image", "image_modification") else max_tokens
            )
            try:
                t_start = time.time()
                inputs = local_lm_processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(local_lm_model.device)
                input_len = inputs["input_ids"].shape[1]
                with torch.no_grad():
                    generated_ids = local_lm_model.generate(
                        **inputs, max_new_tokens=mtokens, do_sample=False
                    )
                gen_tokens = generated_ids[0][input_len:]
                result = local_lm_processor.decode(
                    gen_tokens, skip_special_tokens=True
                ).strip()
                elapsed = time.time() - t_start
            except Exception as e:
                print(f"  Local LM error for zone {idx}: {e}")
                return (idx, z, None)
            z["content"] = result
            z["source"] = "lm_studio"
            _progress[0] += 1
            preview = result[:80].replace("\n", " ")
            try:
                preview = preview.encode("cp1252", errors="replace").decode("cp1252")
            except Exception:
                pass
            print(f"  [{_progress[0]}/{len(lm_tasks)}] Page {z['page']+1} [{z['label']}]: {preview} ({elapsed:.1f}s)")
            return (idx, z, result)

        worker_fn = process_one_local if use_local else process_one_remote
        with ThreadPoolExecutor(max_workers=1 if use_local else args.workers) as pool:
            futures = [pool.submit(worker_fn, t) for t in lm_tasks]
            for f in as_completed(futures):
                f.result()

    if local_lm_model is not None:
        unload_model(local_lm_model, local_lm_processor)
        free_gpu_memory(verbose=True)

    overlay_count = 0
    for z in zones:
        if z.get("source") == "lm_studio" and z.get("content", "").strip():
            overlay_text_on_page(
                doc[z["page"]], z, pdf_dims[z["page"]], invisible=not args.visible
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
