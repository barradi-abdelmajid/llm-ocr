"""End-to-end GLM-OCR pipeline: layout detection + OCR + LM Studio fallback + verification.

Usage:
  # Local mode (default):
  python glmocr_pipeline.py pdf_idk.pdf -o ./output

  # With LM Studio for circuits/images + verification:
  python glmocr_pipeline.py pdf_idk.pdf -o ./output --lm-studio http://100.76.47.104:1234/v1

  # With config file tuning:
  python glmocr_pipeline.py pdf_idk.pdf -o ./output --config config.txt

  # MaaS cloud API mode:
  python glmocr_pipeline.py pdf_idk.pdf --mode maas --api-key sk-xxx
"""

import os, sys, json, time, re, argparse, dataclasses
from pathlib import Path
from typing import List, Optional
import warnings

warnings.filterwarnings("ignore")

import torch
import numpy as np
from PIL import Image

from glmocr.config import LayoutConfig
from glmocr.layout import PPDocLayoutDetector

# ── Constants ──────────────────────────────────────────────────────────────────
TASK_PROMPTS = {
    "text": "Text Recognition:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
}
OBSIDIAN_LABEL_MAP = {
    "text": "text",
    "table": "table",
    "formula": "formula",
    "display_formula": "formula",
    "inline_formula": "formula",
    "doc_title": "title",
    "paragraph_title": "heading",
    "figure_title": "caption",
    "content": "text",
    "abstract": "abstract",
    "algorithm": "text",
    "reference_content": "reference",
    "vertical_text": "text",
    "seal": "seal",
    "formula_number": "formula_num",
    "chart": "chart",
    "image": "image",
}

TASK_ORDER = {
    "circuit": 0,
    "formula": 1,
    "text": 2,
    "image": 3,
    "chart": 4,
    "heading": 5,
    "table": 6,
    "reference": 7,
    "abstract": 8,
    "title": 9,
    "caption": 10,
    "seal": 11,
    "formula_num": 12,
}

LM_STUDIO_PROMPTS = {
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
    "text": (
        "This is a document region that may have OCR errors. "
        "Look at the image carefully and transcribe the text accurately. "
        "Output only the corrected text, nothing else."
    ),
    "table": (
        "Extract the data from this table. "
        "Preserve all rows, columns, and values. "
        "Output as a markdown table."
    ),
    "formula": (
        "Convert this mathematical formula to LaTeX. "
        "Output only the LaTeX code between $$ delimiters."
    ),
    "proofread": (
        "You are a strict markdown proofreader. Compare this PDF page image to its transcription below.\n"
        "Fix ALL of the following issues:\n"
        "1. GRAMMAR: Fix French/English spelling and grammar for human readability\n"
        "2. LATEX: Every equation must use proper $$...$$ (display) or $...$ (inline) delimiters\n"
        "   - Fix broken fractions: replace frac with proper \\\\frac{num}{den}\n"
        "   - Fix broken subscripts/superscripts: ensure _{...} and ^{...} have proper braces\n"
        "   - All LaTeX commands must have correct syntax\n"
        "3. CODE BLOCKS: Ensure all ``` have matching open/close\n"
        "4. ORDERING: Reorder content top-to-bottom, left-to-right to match the PDF\n"
        "5. PRESERVE: Keep all > blockquotes (Mandarin circuit descriptions) exactly as-is\n"
        "Output ONLY the corrected markdown for this page, starting with '## Page N'."
    ),
}


@dataclasses.dataclass
class Zone:
    page: int
    label: str
    content: str
    bbox_norm: List[int]
    source: str = "glm_ocr"


# ── Config Loading ────────────────────────────────────────────────────────────


def load_config(config_path: Optional[str] = None) -> dict:
    """Load settings from a config.txt file. Returns dict of overrides."""
    overrides = {}
    if not config_path:
        return overrides
    path = Path(config_path)
    if not path.exists():
        print(f"  Warning: config file not found: {config_path}")
        return overrides
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, val = line.split("=", 1)
            overrides[key.strip()] = val.strip()
    print(f"  Loaded {len(overrides)} config values from {config_path}")
    return overrides


# ── Layout Detection ─────────────────────────────────────────────────────────


def run_layout_detection(
    page_images: List[Image.Image],
    device: str = "cpu",
    threshold: float = 0.3,
) -> List[List[dict]]:
    """Run PP-DocLayoutV3 on all pages. Returns per-page list of region dicts."""
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
        results, _ = detector.process(page_images, save_visualization=False)
        return results
    finally:
        detector.stop()


# ── GLM-OCR Recognition ─────────────────────────────────────────────────────


def load_glm_ocr():
    """Load GLM-OCR model and processor (cached)."""
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


def recognize_region(
    crop_img: Image.Image,
    task_type: str,
    processor,
    model,
    max_new_tokens: int = 1024,
) -> str:
    """Run GLM-OCR on a cropped region and return recognized text."""
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
    result = processor.decode(gen_tokens, skip_special_tokens=True)
    return result.strip()


# ── Cropping ──────────────────────────────────────────────────────────────────


def crop_region(page_img: Image.Image, bbox_norm: List[int]) -> Image.Image:
    """Convert normalized 0-1000 bbox to pixel coords and crop."""
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


# ── PDF Loading ───────────────────────────────────────────────────────────────


def pdf_to_images(pdf_path: str, dpi: int = 200) -> tuple:
    """Convert PDF pages to PIL Images. Returns (images, pdf_point_dims)."""
    import fitz

    doc = fitz.open(pdf_path)
    images = []
    pdf_dims = []
    for i in range(len(doc)):
        page = doc[i]
        pdf_dims.append((page.rect.width, page.rect.height))
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    doc.close()
    return images, pdf_dims


# ── LM Studio Description / Fallback ─────────────────────────────────────────


def group_circuit_regions(zones: List[Zone]) -> List[List[int]]:
    """Group adjacent image zones likely showing the same circuit.

    Returns list of groups, each group is a list of zone indices into `zones`.
    Consecutive image/chart zones on the same page within 200 normalized Y-units
    are grouped together (first = full circuit, rest = modifications).
    """
    PROXIMITY = 200
    img_indices = [i for i, z in enumerate(zones) if z.label in ("image", "chart")]
    groups = []
    current = []

    for idx in img_indices:
        z = zones[idx]
        if not current:
            current.append(idx)
        else:
            prev_z = zones[current[-1]]
            same_page = z.page == prev_z.page
            prev_cy = (prev_z.bbox_norm[1] + prev_z.bbox_norm[3]) / 2
            curr_cy = (z.bbox_norm[1] + z.bbox_norm[3]) / 2
            if same_page and abs(curr_cy - prev_cy) < PROXIMITY:
                current.append(idx)
            else:
                groups.append(current)
                current = [idx]

    if current:
        groups.append(current)
    return groups


def lm_studio_analyze(
    crop_img: Image.Image,
    prompt_type: str,
    lm_studio_url: str,
    max_tokens: int = 512,
    timeout: int = 60,
) -> str:
    """Send a cropped region to LM Studio for analysis (description, SPICE, etc.)."""
    import requests, base64, io

    buf = io.BytesIO()
    crop_img.save(buf, format="JPEG", quality=95)
    b64 = base64.b64encode(buf.getvalue()).decode()

    prompt = LM_STUDIO_PROMPTS.get(prompt_type, LM_STUDIO_PROMPTS["image"])

    payload = {
        "model": "mistral",
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
        else:
            return f"[LM Studio error: HTTP {resp.status_code}]"
    except requests.exceptions.ConnectionError:
        return "[LM Studio unavailable]"
    except Exception as e:
        return f"[LM Studio error: {e}]"


# ── LM Studio Proofreading + Markdown Syntax Check ──────────────────────────


def lm_studio_proofread_page(
    page_img: Image.Image,
    page_content: str,
    lm_studio_url: str,
    page_num: int,
    max_tokens: int = 2048,
    timeout: int = 120,
) -> str:
    """Send a PDF page + its markdown to LM Studio for proofreading + reordering."""
    import requests, base64, io

    buf = io.BytesIO()
    page_img.save(buf, format="JPEG", quality=95)
    b64 = base64.b64encode(buf.getvalue()).decode()

    payload = {
        "model": "mistral",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                    {
                        "type": "text",
                        "text": f"PDF page transcription:\n\n{page_content}\n\n{LM_STUDIO_PROMPTS['proofread']}",
                    },
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
        return page_content  # return original on error
    except Exception:
        return page_content


def fix_markdown_strict(text: str) -> str:
    """Strict post-processing: remove all code block fences, balance LaTeX delimiters, fix common issues."""
    lines = text.splitlines()
    result = []
    in_code_block = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue

        if in_code_block:
            continue

        if "$$" not in line:
            s_count = line.count("$")
            if s_count % 2 != 0 and s_count > 0:
                line = line + " $"

        line = re.sub(r"\\frac\s+\{", r"\\frac{", line)
        line = re.sub(r"\}\s*\{", r"}{", line)
        line = re.sub(r"\\([a-zA-Z]+)\s+\{", r"\\\1{", line)

        result.append(line)

    full = "\n".join(result)
    dd_count = full.count("$$")
    if dd_count % 2 != 0:
        full += "\n$$"

    return full


def check_markdown_syntax(text: str) -> list:
    """Check markdown for syntax issues. Returns list of warnings."""
    warnings = []

    # Check balanced $$ LaTeX display math
    dollar_count = text.count("$$")
    if dollar_count % 2 != 0:
        warnings.append(f"Unbalanced $$: {dollar_count} delimiters (expected even)")

    # Check $ inline math balance
    lines = text.splitlines()
    for i, line in enumerate(lines, 1):
        s_count = line.count("$")
        if s_count % 2 != 0 and "$$" not in line:
            warnings.append(f"Line {i}: unbalanced $ (odd count)")

    # Check table formatting: header separator must have ---
    for i, line in enumerate(lines, 1):
        if "|" in line and i + 1 <= len(lines):
            nxt = lines[i]  # next line
            if (
                nxt
                and nxt.strip().startswith("|")
                and "---" not in nxt
                and "-" not in nxt
            ):
                # Check if this is actually a table header
                if line.strip().startswith("|") and not line.strip().startswith("|--"):
                    warnings.append(
                        f"Line {i + 1}: table separator may be missing (expect |---|)"
                    )

    return warnings


def verify_zones(
    zones: List[Zone],
    page_count: int,
    pdf_dims: List[tuple],
    pdf_stem: str,
) -> dict:
    """Run verification checks against zones and PDF structure.

    Returns a dict with verification results and flagged issues.
    """
    issues = []

    # Check 1: every page has at least one zone
    pages_with_zones = set(z.page for z in zones)
    for p in range(page_count):
        if p not in pages_with_zones:
            issues.append(f"Page {p + 1}: NO zones detected")

    # Check 2: no zone has empty content (unless it's a seal/formula_num)
    for z in zones:
        skip_empty_check = {"seal", "formula_num", "image", "chart"}
        if not z.content.strip() and z.label not in skip_empty_check:
            issues.append(
                f"Page {z.page + 1} [{z.label}] bbox={z.bbox_norm}: empty content"
            )

    # Check 3: zones are sorted per page (top-to-bottom, left-to-right)
    for p in range(page_count):
        page_zones = [z for z in zones if z.page == p]
        for i in range(len(page_zones) - 1):
            a, b = page_zones[i], page_zones[i + 1]
            a_cy = (a.bbox_norm[1] + a.bbox_norm[3]) / 2
            b_cy = (b.bbox_norm[1] + b.bbox_norm[3]) / 2
            if b_cy < a_cy - 50:  # significant vertical disorder
                issues.append(
                    f"Page {p + 1}: zone order issue — zone {i} ({a.label}) below zone {i + 1} ({b.label}) "
                    f"but has higher Y center ({a_cy:.0f} vs {b_cy:.0f})"
                )

    # Check 4: circuits should have SPICE in content
    for z in zones:
        if z.label == "circuit" and "spice" not in z.content.lower():
            issues.append(
                f"Page {z.page + 1} [circuit] bbox={z.bbox_norm}: missing SPICE netlist"
            )

    return {
        "total_zones": len(zones),
        "pages_covered": len(pages_with_zones),
        "pages_total": page_count,
        "issue_count": len(issues),
        "issues": issues,
        "passed": len(issues) == 0,
    }


def write_verification_report(verification: dict, output_dir: str):
    """Write verification report to file."""
    lines = ["# Verification Report", ""]
    lines.append(
        f"**Status:** {'PASSED' if verification['passed'] else 'ISSUES FOUND'}"
    )
    lines.append(f"**Zones:** {verification['total_zones']}")
    lines.append(
        f"**Pages:** {verification['pages_covered']}/{verification['pages_total']} covered"
    )
    lines.append(f"**Issues:** {verification['issue_count']}")
    lines.append("")

    if verification["issues"]:
        lines.append("## Issues")
        for issue in verification["issues"]:
            lines.append(f"- {issue}")
        lines.append("")

    report = "\n".join(lines)
    Path(output_dir, "verification.md").write_text(report, encoding="utf-8")
    return report


# ── Output Formatting ─────────────────────────────────────────────────────────


def sort_zones_on_page(zones: List[Zone]) -> List[Zone]:
    """Sort zones within a page: top-to-bottom, then left-to-right, grouped by type."""
    return sorted(
        zones,
        key=lambda z: (
            TASK_ORDER.get(z.label, 99),
            z.bbox_norm[1],  # y1 (top first)
            z.bbox_norm[0],  # x1 (left first)
        ),
    )


def format_obsidian(
    zones: List[Zone],
    pdf_stem: str,
    pdf_dims: List[tuple],
) -> str:
    """Format zones as Obsidian markdown with PDF embed links."""
    pdf_name = f"{pdf_stem}.pdf"
    lines = []

    # Group and sort by page
    for page_idx in range(len(pdf_dims)):
        page_zones = [z for z in zones if z.page == page_idx]
        page_zones = sort_zones_on_page(page_zones)
        pw, ph = pdf_dims[page_idx]

        for z in page_zones:
            l = int(z.bbox_norm[0] * pw / 1000)
            b = int((1000 - z.bbox_norm[3]) * ph / 1000)
            r = int(z.bbox_norm[2] * pw / 1000)
            t = int((1000 - z.bbox_norm[1]) * ph / 1000)

            label = OBSIDIAN_LABEL_MAP.get(z.label, z.label)
            link = f"![[{pdf_name}#page={z.page + 1}&rect={l},{b},{r},{t}|{label}]]"
            lines.append(link)
            if z.content.strip():
                lines.append(z.content.strip())
            lines.append("")

    return "\n".join(lines)


def format_final_markdown(
    zones: List[Zone],
    pdf_dims: List[tuple],
) -> str:
    """Produce clean markdown with all content but no Obsidian ![[...]] embed links."""
    lines = []
    for page_idx in range(len(pdf_dims)):
        page_zones = [z for z in zones if z.page == page_idx]
        if not page_zones:
            continue
        page_zones = sort_zones_on_page(page_zones)
        lines.append(f"## Page {page_idx + 1}")
        lines.append("")
        for z in page_zones:
            content = z.content.strip()
            if content:
                content_lines = content.splitlines()
                content_lines = [
                    cl for cl in content_lines if not cl.strip().startswith("```")
                ]
                content = "\n".join(content_lines).strip()
                if content:
                    lines.append(content)
                    lines.append("")
    return "\n".join(lines)


def format_llm_report(zones: List[Zone], pdf_filename: str) -> str:
    """Format zone report for the LLM to read."""
    lines = [f"# Zone Report: {pdf_filename}", ""]

    for page_idx in sorted(set(z.page for z in zones)):
        page_zones = [z for z in zones if z.page == page_idx]
        page_zones = sort_zones_on_page(page_zones)

        lines.append(f"## Page {page_idx + 1}")
        lines.append("")

        for z in page_zones:
            lines.append(f"### [{z.label}] (source: {z.source})")
            lines.append(f"bbox: {z.bbox_norm}")
            if z.content.strip():
                lines.append(z.content.strip())
            lines.append("")

    return "\n".join(lines)


# ── Cloud API Mode ────────────────────────────────────────────────────────────


def run_maas_mode(pdf_path: str, api_key: str, output_dir: str):
    """Run GLM-OCR via cloud API (MaaS)."""
    import glmocr
    from glmocr import GlmOcr

    pdf_path = str(Path(pdf_path).resolve())
    pdf_name = Path(pdf_path).stem

    parser = GlmOcr(api_key=api_key, mode="maas")
    try:
        result = parser.parse(pdf_path)
        zones = []
        page_dims = []

        for page_idx, page_regions in enumerate(result.json_result):
            page_dims.append((1000, 1000))
            for reg in page_regions:
                zones.append(
                    Zone(
                        page=page_idx,
                        label=reg.get("label", "text"),
                        content=reg.get("content", "") or "",
                        bbox_norm=reg.get("bbox_2d", [0, 0, 0, 0]),
                        source="maas",
                    )
                )

        obsidian = format_obsidian(zones, pdf_name, page_dims)
        report = format_llm_report(zones, pdf_name)

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "output.md").write_text(obsidian, encoding="utf-8")
        (out / "llm_output.md").write_text(report, encoding="utf-8")

        print(f"  Saved {len(zones)} zones to {output_dir}/")
        return zones
    finally:
        parser.close()


# ── Main Pipeline ─────────────────────────────────────────────────────────────


def run_local_pipeline(
    pdf_path: str,
    output_dir: str = ".",
    layout_device: str = "cpu",
    lm_studio_url: Optional[str] = None,
    config: Optional[dict] = None,
    skip_verification: bool = False,
):
    """Full pipeline: layout + GLM-OCR + LM Studio fallback + verification."""
    pdf_path = str(Path(pdf_path).resolve())
    pdf_name = Path(pdf_path).stem
    config = config or {}
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Convert PDF to images
    print(f"[1/7] Converting PDF to images...")
    t0 = time.time()
    dpi = int(config.get("PDF_DPI", 200))
    page_images, pdf_dims = pdf_to_images(pdf_path, dpi=dpi)
    print(f"  {len(page_images)} pages in {time.time() - t0:.1f}s")

    # Step 2: Layout detection
    print(f"[2/7] Running layout detection (PP-DocLayoutV3)...")
    t0 = time.time()
    layout_results = run_layout_detection(
        page_images,
        device=config.get("LAYOUT_DEVICE", layout_device),
        threshold=float(config.get("LAYOUT_THRESHOLD", 0.3)),
    )
    total_regions = sum(len(r) for r in layout_results)
    print(f"  {total_regions} regions detected in {time.time() - t0:.1f}s")

    # Step 3: GLM-OCR for text / formula / table regions
    print(f"[3/7] Running GLM-OCR recognition...")
    processor, model = load_glm_ocr()
    zones = []
    t0 = time.time()
    try:
        for page_idx, regions in enumerate(layout_results):
            for region in regions:
                task_type = region["task_type"]
                bbox = region["bbox_2d"]

                if task_type == "skip":
                    # Image/chart — defer to LM Studio later
                    zones.append(
                        Zone(
                            page=page_idx,
                            label=region["label"],
                            content="",
                            bbox_norm=bbox,
                            source="pending",
                        )
                    )
                    continue

                crop_img = crop_region(page_images[page_idx], bbox)
                text = recognize_region(crop_img, task_type, processor, model)
                zones.append(
                    Zone(
                        page=page_idx,
                        label=region["label"],
                        content=text,
                        bbox_norm=bbox,
                        source="glm_ocr",
                    )
                )
                preview = text[:80].replace("\n", " ")
                print(f"  Page {page_idx + 1} [{region['label']}]: {preview}")
    finally:
        del processor, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f"  {len(zones)} regions recognized in {time.time() - t0:.1f}s")

    # Step 4: LM Studio description for images/charts + GLM-OCR fallback
    lm_url = lm_studio_url or config.get("LM_STUDIO_HOST")
    if lm_url:
        # Quick connectivity test
        lm_available = False
        try:
            import requests

            r = requests.get(f"{lm_url.rstrip('/')}/models", timeout=5)
            lm_available = r.status_code == 200
        except Exception:
            lm_available = False

        if not lm_available:
            print(f"[4/7] LM Studio not reachable at {lm_url}, skipping")
            for z in zones:
                if z.source == "pending":
                    z.source = "skipped"
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            print(f"[4/7] Running LM Studio analysis (fallback & description)...")
            t0 = time.time()

            lm_tasks = []

            # Group circuit images so first gets full description, rest get modification
            circuit_groups = group_circuit_regions(zones)
            # Build a set of (index -> prompt_type) for image zones
            prompt_overrides = {}
            for group in circuit_groups:
                for i, idx in enumerate(group):
                    prompt_overrides[idx] = "image" if i == 0 else "image_modification"

            for i, z in enumerate(zones):
                needs_lm = False
                prompt_type = z.label

                if z.label in ("image", "chart"):
                    needs_lm = True
                    prompt_type = prompt_overrides.get(i, "image")
                elif z.label in (
                    "text",
                    "content",
                    "paragraph_title",
                    "reference_content",
                ):
                    if not z.content.strip():
                        needs_lm = True
                        prompt_type = "text"
                    elif z.source == "pending":
                        needs_lm = True
                        prompt_type = "text"
                elif z.source == "pending":
                    needs_lm = True
                    prompt_type = z.label

                if needs_lm:
                    crop_img = crop_region(page_images[z.page], z.bbox_norm)
                    lm_tasks.append((z, crop_img, prompt_type))

            def process_one(args):
                z, crop, ptype = args
                original = z.content
                mtokens = 128 if ptype in ("image", "image_modification") else 512
                result = lm_studio_analyze(crop, ptype, lm_url, max_tokens=mtokens)
                if result.startswith("[LM Studio"):
                    if original.strip():
                        z.content = original
                    z.source = "glm_ocr"
                    return (z, f"LM unavailable, kept original")
                z.content = result
                z.source = "lm_studio"
                return (z, result[:80].replace("\n", " "))

            lm_zones = 0
            with ThreadPoolExecutor(max_workers=5) as pool:
                futures = [pool.submit(process_one, t) for t in lm_tasks]
                for f in as_completed(futures):
                    z, preview = f.result()
                    lm_zones += 1
                    print(f"  LM Page {z.page + 1} [{z.label}]: {preview}")

            print(
                f"  {lm_zones} regions analyzed by LM Studio in {time.time() - t0:.1f}s"
            )

            # Save intermediate output (in case pipeline times out)
            obsidian = format_obsidian(zones, pdf_name, pdf_dims)
            Path(output_dir, "output.md").write_text(obsidian, encoding="utf-8")
            report = format_llm_report(zones, pdf_name)
            Path(output_dir, "llm_output.md").write_text(report, encoding="utf-8")
            final = format_final_markdown(zones, pdf_dims)
            Path(output_dir, "final.md").write_text(final, encoding="utf-8")
            print(f"  Intermediate save -> {output_dir}/")
    else:
        print(f"[4/7] Skipping LM Studio (no --lm-studio URL provided)")
        for z in zones:
            if z.source == "pending":
                z.source = "skipped"

    # Step 5: LM Studio page-by-page proofreading + reordering
    if lm_url and lm_available:
        print(f"[5/7] Running page-by-page proofreading...")
        t0 = time.time()
        final_md = format_final_markdown(zones, pdf_dims)
        pages_md = final_md.split("## Page ")

        corrected_pages = []
        for pi in range(len(page_images)):
            page_header = f"## Page {pi + 1}"
            # Find this page's content in the split
            page_content = ""
            for chunk in pages_md:
                if chunk.strip().startswith(str(pi + 1)):
                    page_content = (
                        page_header + "\n" + "\n".join(chunk.splitlines()[1:])
                    )
                    break

            if not page_content.strip():
                corrected_pages.append(page_content)
                continue

            result = lm_studio_proofread_page(
                page_images[pi], page_content, lm_url, pi + 1
            )
            corrected_pages.append(result)
            preview = result[:60].replace("\n", " ")
            print(f"  Proofread page {pi + 1}: {preview}...")

        proofread_final = "\n\n".join(cp for cp in corrected_pages if cp.strip())
        Path(output_dir, "final.md").write_text(proofread_final, encoding="utf-8")
        print(f"  Proofread {len(page_images)} pages in {time.time() - t0:.1f}s")

        # Also update output.md with corrected content (regenerate from corrected pages)
        # Re-read the proofread final and update zones content for output.md
        # For simplicity, just regenerate output.md from zones (proofread only touches final.md)
    else:
        # Still generate final.md from zones
        final_md = format_final_markdown(zones, pdf_dims)
        Path(output_dir, "final.md").write_text(final_md, encoding="utf-8")
        print(f"  Saved final.md (no proofreading)")

    # Step 6: Verification
    print(f"[6/7] Running verification...")
    verification = verify_zones(zones, len(page_images), pdf_dims, pdf_name)
    write_verification_report(verification, output_dir)
    if verification["passed"]:
        print(f"  Verification: PASSED ({verification['total_zones']} zones)")
    else:
        print(f"  Verification: {verification['issue_count']} issue(s) found")
        for issue in verification["issues"][:10]:
            print(f"    - {issue}")
        if verification["issue_count"] > 10:
            print(f"    ... and {verification['issue_count'] - 10} more")

    # Step 7: Markdown syntax fix + check
    print(f"[7/7] Fixing markdown syntax...")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    final_path = out / "final.md"
    if final_path.exists():
        raw = final_path.read_text(encoding="utf-8")
        fixed = fix_markdown_strict(raw)
        final_path.write_text(fixed, encoding="utf-8")
        fixed_count = 1 if raw != fixed else 0

        md_warnings = check_markdown_syntax(fixed)
        if md_warnings:
            print(f"  Markdown syntax warnings ({len(md_warnings)}):")
            for w in md_warnings:
                print(f"    - {w}")
        elif fixed_count:
            print(f"  Markdown: fixed, syntax OK")
        else:
            print(f"  Markdown syntax: OK")

    # Regenerate output.md and llm_output.md from zones
    obsidian = format_obsidian(zones, pdf_name, pdf_dims)
    (out / "output.md").write_text(obsidian, encoding="utf-8")
    print(f"  Saved output.md")

    report = format_llm_report(zones, pdf_name)
    (out / "llm_output.md").write_text(report, encoding="utf-8")
    print(f"  Saved llm_output.md")

    print(f"  Total: {len(zones)} zones -> {output_dir}/")

    return zones, verification


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="GLM-OCR Pipeline v2")
    parser.add_argument("pdf", help="Path to PDF file")
    parser.add_argument(
        "--output-dir", "-o", default="./output", help="Output directory"
    )
    parser.add_argument(
        "--mode",
        choices=["local", "maas"],
        default="local",
        help="local (transformers) or maas (cloud API)",
    )
    parser.add_argument("--api-key", help="ZHIPU_API_KEY for MaaS mode")
    parser.add_argument(
        "--layout-device", default="cpu", help="Device for layout detection"
    )
    parser.add_argument(
        "--lm-studio",
        help="LM Studio API URL (e.g. http://100.76.47.104:1234/v1)",
    )
    parser.add_argument(
        "--config",
        default="config.txt",
        help="Path to config.txt for fine-tuning parameters",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.pdf):
        print(f"Error: PDF not found: {args.pdf}")
        sys.exit(1)

    # Load config file
    config = load_config(args.config)

    # Merge CLI overrides into config
    if args.lm_studio:
        config["LM_STUDIO_HOST"] = args.lm_studio

    print(f"GLM-OCR Pipeline v2")
    print(f"  PDF:       {args.pdf}")
    print(f"  Mode:      {args.mode}")
    print(f"  Output:    {args.output_dir}")
    if config.get("LM_STUDIO_HOST"):
        print(f"  LM Studio: {config['LM_STUDIO_HOST']}")

    t_total = time.time()

    if args.mode == "maas":
        if not args.api_key:
            print("Error: --api-key required for MaaS mode")
            sys.exit(1)
        run_maas_mode(args.pdf, args.api_key, args.output_dir)
    else:
        run_local_pipeline(
            args.pdf,
            output_dir=args.output_dir,
            layout_device=args.layout_device,
            lm_studio_url=config.get("LM_STUDIO_HOST"),
            config=config,
        )

    print(f"\nTotal time: {time.time() - t_total:.1f}s")
    print(f"Done. Output in {args.output_dir}/")


if __name__ == "__main__":
    main()
