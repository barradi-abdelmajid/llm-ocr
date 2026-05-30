"""LM Studio-based layout detection (cross-platform alternative to PP-DocLayoutV3).
Sends each page image to LM Studio vision model for region identification.
Returns same format as glmocr.layout.PPDocLayoutDetector.process().
"""

import os, json, time, io, base64, re
from pathlib import Path
from PIL import Image
import requests


SYSTEM_PROMPT = (
    "You are a precise document layout analyzer. "
    "Identify all distinct content regions on the page and output as JSON only."
)

USER_PROMPT = (
    "List every distinct content region visible on this page. "
    "For each region, specify:\n"
    '  - "label": the most specific type from: text, title, subtitle, paragraph, list, table, formula, inline_formula, image, chart, figure, header, footer, page_number, caption, figure_title, table_title\n'
    '  - "bbox": [x1, y1, x2, y2] where coordinates are in 0-1000 scale relative to page width/height from top-left\n\n'
    "Rules:\n"
    "- Cover ALL visible content. Split multi-column layouts into separate regions.\n"
    "- Each formula or inline formula is its own region.\n"
    "- Tables are single regions.\n"
    "- Images and charts get their own regions.\n"
    "- Headers/footers/page numbers each get their own region if present.\n"
    "- Do NOT merge text and formulas into one region.\n"
    "- Be generous with coordinates — slightly over-estimate is better than under-estimate.\n\n"
    "Output ONLY a valid JSON array, no other text before or after:\n"
    '[{"label":"text","bbox":[100,100,500,200]}, ...]'
)


def _call_lm_studio(page_img: Image.Image, lm_url: str, model: str, timeout: int = 60) -> list:
    buf = io.BytesIO()
    page_img.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode()

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": USER_PROMPT},
                ],
            },
        ],
        "max_tokens": 2048,
        "temperature": 0.0,
    }

    resp = requests.post(
        f"{lm_url.rstrip('/')}/chat/completions",
        json=payload,
        timeout=timeout,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"LM Studio HTTP {resp.status_code}: {resp.text[:200]}")

    text = resp.json()["choices"][0]["message"]["content"].strip()

    # Try to extract JSON array from the response
    json_match = re.search(r"\[.*?\]", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # Try parsing the whole response as JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    raise RuntimeError(f"Could not parse LM Studio layout output: {text[:300]}")


# Mapping from LM Studio label names → task_type (same as PP-DocLayoutV3)
LABEL_TO_TASK = {
    # text group
    "text": "text",
    "title": "text",
    "subtitle": "text",
    "paragraph": "text",
    "list": "text",
    "caption": "text",
    "figure_title": "text",
    "table_title": "text",
    "abstract": "text",
    "reference": "abandon",
    # formula group
    "formula": "formula",
    "inline_formula": "formula",
    # table
    "table": "table",
    # skip group (images/charts → VLM analysis in Phase 3)
    "image": "skip",
    "chart": "skip",
    "figure": "skip",
    # abandon
    "header": "abandon",
    "footer": "abandon",
    "page_number": "abandon",
}


def run_lm_layout_detection(page_images: list, lm_url: str, model: str,
                            timeout: int = 60, dpi: int = 200) -> tuple:
    """Run LM Studio layout detection on page images.

    Returns:
        Tuple of (results, vis_images) matching PPDocLayoutDetector.process() format:
          - results: List[List[Dict]] — per-page list of region dicts with keys:
            index, label, score, bbox_2d, polygon, task_type
          - vis_images: Dict[int, Image.Image] — visualization images (empty for now)
    """
    all_results = []
    errors = []

    for page_idx, img in enumerate(page_images):
        t_start = time.time()
        try:
            regions = _call_lm_studio(img, lm_url, model, timeout=timeout)
        except Exception as e:
            errors.append(f"  Page {page_idx+1}: {e}")
            all_results.append([])
            continue

        page_results = []
        for i, r in enumerate(regions):
            if not isinstance(r, dict):
                continue
            label = str(r.get("label", "text")).lower().strip()
            bbox = r.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = bbox
            # Clamp to 0-1000
            x1 = max(0, min(1000, int(x1)))
            y1 = max(0, min(1000, int(y1)))
            x2 = max(0, min(1000, int(x2)))
            y2 = max(0, min(1000, int(y2)))
            if x2 <= x1 or y2 <= y1:
                continue

            task_type = LABEL_TO_TASK.get(label, "text")

            page_results.append({
                "index": i,
                "label": label,
                "score": float(r.get("score", 1.0)),
                "bbox_2d": [x1, y1, x2, y2],
                "polygon": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                "task_type": task_type,
            })

        elapsed = time.time() - t_start
        print(f"    Page {page_idx+1}: {len(page_results)} regions in {elapsed:.1f}s")
        all_results.append(page_results)

    for e in errors:
        print(e)

    return all_results, {}
