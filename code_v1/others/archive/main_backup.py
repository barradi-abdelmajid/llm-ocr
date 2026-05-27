"""
Phase 1: Zoning with categorization.
Extracts Notability text zones + embedded images + ink-based drawing regions.
Verifies all ink is covered by zones, flags any missed regions.
Optionally validates zone quality with LM Studio VLM.
Outputs Obsidian PDF embeds to output.md for visual debugging.
"""

import sys, io, base64, os, tempfile

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import fitz
import numpy as np
from docling.document_converter import DocumentConverter
from collections import Counter
from scipy import ndimage

PDF_PATH = "pdf_idk.pdf"
OUTPUT_PATH = "output.md"
LLM_OUTPUT_PATH = "llm_output.md"
PDF_FILENAME = "pdf_idk.pdf"
MIN_AREA = 50
Y_GAP = 18
INK_THRESHOLD = 220
MIN_DRAWING_PIXELS = 80

LM_STUDIO_HOST = "http://100.76.47.104:1234"
LM_VERIFY = True  # set True to run gap analysis via LM Studio after zoning


def extract_notability_objects(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for pn in range(doc.page_count):
        page = doc[pn]
        ph = page.rect.height
        blocks = page.get_text("dict")["blocks"]
        objs = []
        for b in blocks:
            if b["type"] != 0:
                continue
            for line in b["lines"]:
                x0, y0, x1, y1 = line["bbox"]
                area = (x1 - x0) * (y1 - y0)
                if area < MIN_AREA:
                    continue
                objs.append(
                    {
                        "x0": x0,
                        "x1": x1,
                        "y_bot": ph - y1,
                        "y_top": ph - y0,
                        "cy_pdf": ph - (y0 + y1) / 2,
                    }
                )
        pages.append({"num": pn + 1, "objects": objs, "height": ph})
    doc.close()
    return pages


def extract_embedded_images(pdf_path, pages):
    doc = fitz.open(pdf_path)
    for pn in range(doc.page_count):
        page = doc[pn]
        blocks = page.get_text("dict")["blocks"]
        pw, ph = page.rect.width, page.rect.height
        images = []
        for b in blocks:
            if b["type"] != 1:
                continue
            x0, y0, x1, y1 = b["bbox"]
            bw, bh = x1 - x0, y1 - y0
            if bw > pw * 0.9 and bh > ph * 0.9:
                continue
            images.append({"x0": x0, "x1": x1, "y_bot": ph - y1, "y_top": ph - y0})
        pages[pn]["images"] = images
    doc.close()


def find_ink_components(pdf_path, pages):
    doc = fitz.open(pdf_path)
    for pn in range(doc.page_count):
        page = doc[pn]
        pix = page.get_pixmap(dpi=72)
        pw, ph = page.rect.width, page.rect.height
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        gray = img[..., :3].mean(axis=2) if pix.n >= 3 else img[..., 0]
        ink = gray < INK_THRESHOLD

        labeled, n_features = ndimage.label(ink)
        components = []
        for fid in range(1, n_features + 1):
            ys, xs = np.where(labeled == fid)
            px0, py0, px1, py1 = xs.min(), ys.min(), xs.max(), ys.max()
            n = len(ys)
            peri_count = ((xs == px0) | (xs == px1) | (ys == py0) | (ys == py1)).sum()
            components.append(
                {
                    "x0": px0 / pix.width * pw,
                    "x1": px1 / pix.width * pw,
                    "y_bot": ph - py1 / pix.height * ph,
                    "y_top": ph - py0 / pix.height * ph,
                    "pixels": n,
                    "peri_ratio": peri_count / n if n > 0 else 0,
                    "bw_px": px1 - px0,
                    "bh_px": py1 - py0,
                }
            )
        pages[pn]["ink_components"] = components
    doc.close()


def find_drawing_regions(pages):
    for pn, page in enumerate(pages):
        occupied = list(page.get("objects", []))
        for imgz in page.get("images", []):
            occupied.append(imgz)

        zones = []
        for comp in page.get("ink_components", []):
            if comp["pixels"] < MIN_DRAWING_PIXELS:
                continue
            n = comp["pixels"]
            w_px = comp.get("bw_px", 0)
            h_px = comp.get("bh_px", 0)
            bbox_area_px = max(1, w_px * h_px)
            density = n / bbox_area_px
            is_sparse = density < 0.3 and comp["peri_ratio"] > 0.15

            overlaps = False
            for o in occupied:
                if overlap_ratio(comp, o) > 0.7:
                    overlaps = True
                    break
            if not overlaps or is_sparse:
                zones.append(
                    {
                        "x0": comp["x0"],
                        "x1": comp["x1"],
                        "y_bot": comp["y_bot"],
                        "y_top": comp["y_top"],
                    }
                )
        pages[pn]["drawings"] = zones


def find_box_zones(pdf_path, pages):
    doc = fitz.open(pdf_path)
    for pn, page in enumerate(pages):
        p = doc[pn]
        pix = p.get_pixmap(dpi=72)
        pw, ph = p.rect.width, p.rect.height
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        gray = img[..., :3].mean(axis=2) if pix.n >= 3 else img[..., 0]
        ink = gray < INK_THRESHOLD

        holes = ~ink
        labeled, n_features = ndimage.label(holes)
        h, w = holes.shape

        border_ids = set()
        border_ids.update(labeled[0, :].tolist())
        border_ids.update(labeled[-1, :].tolist())
        border_ids.update(labeled[:, 0].tolist())
        border_ids.update(labeled[:, -1].tolist())

        boxes = []
        for fid in range(1, n_features + 1):
            if fid in border_ids:
                continue
            ys, xs = np.where(labeled == fid)
            if len(ys) < 500:
                continue
            px0, py0, px1, py1 = xs.min(), ys.min(), xs.max(), ys.max()
            bw, bh = px1 - px0, py1 - py0
            if bw < 40 or bh < 40 or bw > 500 or bh > 500:
                continue
            if bw < bh * 0.3 or bh < bw * 0.3:  # must be roughly square
                continue

            hole_area = len(ys)
            bbox_area = bw * bh
            if hole_area / bbox_area < 0.5:
                continue

            # check each of 4 sides independently — all must have ink
            margin = max(2, min(bw, bh) // 20)
            ox0, oy0 = max(0, px0 - margin), max(0, py0 - margin)
            ox1, oy1 = min(w - 1, px1 + margin), min(h - 1, py1 + margin)

            top_strip = ink[oy0 : py0 + 1, ox0 : ox1 + 1]
            bot_strip = ink[py1 : oy1 + 1, ox0 : ox1 + 1]
            lft_strip = ink[oy0 : oy1 + 1, ox0 : px0 + 1]
            rgt_strip = ink[oy0 : oy1 + 1, px1 : ox1 + 1]

            # each side must have at least some ink pixels
            min_ink = max(8, margin * 2)
            if any(
                s.sum() < min_ink for s in [top_strip, bot_strip, lft_strip, rgt_strip]
            ):
                continue

            # no overlap with full-page images
            skip = False
            for imgz in page.get("images", []):
                if (imgz["x1"] - imgz["x0"]) * (imgz["y_top"] - imgz["y_bot"]) > 50000:
                    cx0 = px0 / pix.width * pw
                    cy0 = ph - py1 / pix.height * ph
                    cx1 = px1 / pix.width * pw
                    cy1 = ph - py0 / pix.height * ph
                    ix = max(cx0, imgz["x0"]) < min(cx1, imgz["x1"])
                    iy = max(cy0, imgz["y_bot"]) < min(cy1, imgz["y_top"])
                    if ix and iy:
                        skip = True
                        break
            if skip:
                continue

            margin_pt = 10
            boxes.append(
                {
                    "x0": max(0, px0 / pix.width * pw - margin_pt),
                    "x1": min(pw, px1 / pix.width * pw + margin_pt),
                    "y_bot": max(0, ph - py1 / pix.height * ph - margin_pt),
                    "y_top": min(ph, ph - py0 / pix.height * ph + margin_pt),
                    "label": "encadre",
                }
            )
        pages[pn]["boxes"] = boxes
    doc.close()


def y_cluster_pdf(objects, gap=Y_GAP):
    if not objects:
        return []
    sorted_objs = sorted(objects, key=lambda o: o["cy_pdf"])
    clusters = []
    cur = [sorted_objs[0]]
    for obj in sorted_objs[1:]:
        if abs(obj["cy_pdf"] - cur[-1]["cy_pdf"]) <= gap:
            cur.append(obj)
        else:
            clusters.append(cur)
            cur = [obj]
    clusters.append(cur)
    return clusters


def get_docling_layout(pdf_path):
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    doc = result.document
    items = []
    for item in doc.texts:
        label = item.label.value
        for prov in item.prov:
            items.append(
                {
                    "page": prov.page_no,
                    "label": label,
                    "x0": prov.bbox.l,
                    "x1": prov.bbox.r,
                    "y_bot": prov.bbox.b,
                    "y_top": prov.bbox.t,
                }
            )
    return items


def classify(cluster, page_items):
    cx0 = min(o["x0"] for o in cluster)
    cx1 = max(o["x1"] for o in cluster)
    cy_bot = min(o["y_bot"] for o in cluster)
    cy_top = max(o["y_top"] for o in cluster)

    scores = Counter()
    for di in page_items:
        ox0 = max(cx0, di["x0"])
        ox1 = min(cx1, di["x1"])
        oy_bot = max(cy_bot, di["y_bot"])
        oy_top = min(cy_top, di["y_top"])
        if ox0 < ox1 and oy_bot < oy_top:
            overlap = (ox1 - ox0) * (oy_top - oy_bot)
            scores[di["label"]] += overlap
    return scores.most_common(1)[0][0] if scores else "text"


def guess_drawing_label(w, h):
    ratio = w / h if h > 0 else 0
    if 0.5 < ratio < 2.0:
        return "circuit" if h < 300 else "drawing"
    return "drawing"


def overlap_ratio(inner, outer):
    ox0 = max(inner["x0"], outer["x0"])
    ox1 = min(inner["x1"], outer["x1"])
    oy_bot = max(inner["y_bot"], outer["y_bot"])
    oy_top = min(inner["y_top"], outer["y_top"])
    if ox0 < ox1 and oy_bot < oy_top:
        overlap = (ox1 - ox0) * (oy_top - oy_bot)
        inner_area = (inner["x1"] - inner["x0"]) * (inner["y_top"] - inner["y_bot"])
        return overlap / inner_area if inner_area > 0 else 0
    return 0


def zones_overlap(a, b, min_ratio=0.08):
    ox0 = max(a["x0"], b["x0"])
    ox1 = min(a["x1"], b["x1"])
    oy_bot = max(a["y_bot"], b["y_bot"])
    oy_top = min(a["y_top"], b["y_top"])
    if ox0 < ox1 and oy_bot < oy_top:
        overlap_area = (ox1 - ox0) * (oy_top - oy_bot)
        a_area = (a["x1"] - a["x0"]) * (a["y_top"] - a["y_bot"])
        b_area = (b["x1"] - b["x0"]) * (b["y_top"] - b["y_bot"])
        smaller = min(a_area, b_area)
        return overlap_area / smaller >= min_ratio if smaller > 0 else False
    return False


def merge_union(a, b, margin_pt=6):
    return {
        "x0": max(0, min(a["x0"], b["x0"]) - margin_pt),
        "x1": max(a["x1"], b["x1"]) + margin_pt,
        "y_bot": max(0, min(a["y_bot"], b["y_bot"]) - margin_pt),
        "y_top": max(a["y_top"], b["y_top"]) + margin_pt,
        "label": a.get("label", "text"),
    }


def merge_overlapping(zones):
    if not zones:
        return []
    kept = list(zones)
    changed = True
    while changed:
        changed = False
        new_kept = []
        merged = set()
        for i in range(len(kept)):
            if i in merged:
                continue
            cur = dict(kept[i])
            for j in range(i + 1, len(kept)):
                if j in merged:
                    continue
                if zones_overlap(cur, kept[j]):
                    cur = merge_union(cur, kept[j])
                    merged.add(j)
                    changed = True
            new_kept.append(cur)
        kept = new_kept
    return kept


def merge_proximate(zones, xy_gap=12):
    """Merge zones whose expanded bboxes overlap (catch nearby strokes)."""
    if not zones:
        return []
    kept = list(zones)
    changed = True
    while changed:
        changed = False
        new_kept = []
        merged = set()
        for i in range(len(kept)):
            if i in merged:
                continue
            cur = dict(kept[i])
            cx0 = cur["x0"] - xy_gap
            cx1 = cur["x1"] + xy_gap
            cy_bot = cur["y_bot"] - xy_gap
            cy_top = cur["y_top"] + xy_gap
            for j in range(i + 1, len(kept)):
                if j in merged:
                    continue
                n = kept[j]
                if (
                    cx0 < n["x1"]
                    and cx1 > n["x0"]
                    and cy_bot < n["y_top"]
                    and cy_top > n["y_bot"]
                ):
                    cur = merge_union(cur, n, margin_pt=4)
                    merged.add(j)
                    changed = True
                    cx0 = cur["x0"] - xy_gap
                    cx1 = cur["x1"] + xy_gap
                    cy_bot = cur["y_bot"] - xy_gap
                    cy_top = cur["y_top"] + xy_gap
            new_kept.append(cur)
        kept = new_kept
    return kept


def remove_contained(zones):
    """Remove zones fully inside another (keep the outer)."""
    kept = list(zones)
    changed = True
    while changed:
        changed = False
        removed = set()
        for i in range(len(kept)):
            if i in removed:
                continue
            for j in range(len(kept)):
                if i == j or j in removed:
                    continue
                a, b = kept[i], kept[j]
                if (
                    a["x0"] >= b["x0"] - 2
                    and a["x1"] <= b["x1"] + 2
                    and a["y_bot"] >= b["y_bot"] - 2
                    and a["y_top"] <= b["y_top"] + 2
                ):
                    removed.add(j)
                    changed = True
        kept = [z for idx, z in enumerate(kept) if idx not in removed]
    return kept


def find_uncovered_ink(pdf_path, page_num, zones, page_height, min_pixels=15):
    import fitz as fz

    doc = fz.open(pdf_path)
    page = doc[page_num - 1]
    pix = page.get_pixmap(dpi=72)
    pw, ph = page.rect.width, page.rect.height
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n
    )
    gray = img[..., :3].mean(axis=2) if pix.n >= 3 else img[..., 0]
    ink = gray < INK_THRESHOLD

    covered = np.zeros_like(ink, dtype=bool)
    for z in zones:
        x0 = max(0, int(z["x0"] / pw * pix.width))
        x1 = min(pix.width, int(z["x1"] / pw * pix.width) + 1)
        y0 = max(0, int((ph - z["y_top"]) / ph * pix.height))
        y1 = min(pix.height, int((ph - z["y_bot"]) / ph * pix.height) + 1)
        covered[y0:y1, x0:x1] = True

    uncovered_ink = ink & ~covered
    labeled, n_features = ndimage.label(uncovered_ink)
    blobs = []
    for fid in range(1, n_features + 1):
        ys, xs = np.where(labeled == fid)
        if len(ys) < min_pixels:
            continue
        blobs.append(
            {
                "x0": xs.min() / pix.width * pw,
                "x1": xs.max() / pix.width * pw,
                "y_bot": ph - ys.max() / pix.height * ph,
                "y_top": ph - ys.min() / pix.height * ph,
                "pixels": len(ys),
            }
        )
    doc.close()
    return blobs


# --- LM Studio VLM verification ---


def check_lm_studio():
    import httpx

    try:
        r = httpx.post(
            f"{LM_STUDIO_HOST}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 2,
            },
            timeout=10,
        )
        body = r.json()
        loaded = body.get("model", "unknown") if r.status_code == 200 else "unknown"
        print(f"  LM Studio reachable (loaded: {loaded})")
        return True
    except Exception as e:
        print(f"  LM Studio not reachable ({e})")
        return False


def crop_zone(pdf_path, page_num, zone, zoom=1):
    doc = fitz.open(pdf_path)
    page = doc[page_num - 1]
    ph = page.rect.height
    clip = fitz.Rect(zone["x0"], ph - zone["y_top"], zone["x1"], ph - zone["y_bot"])
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip)
    doc.close()
    return pix


def verify_zone(pdf_path, page_num, zone, label, attempt=1):
    import httpx, time

    pix = crop_zone(pdf_path, page_num, zone)
    img_b64 = base64.b64encode(pix.tobytes("png")).decode()

    prompt = (
        "You verify PDF zone segmentation quality."
        f" Label: {label}."
        " Reply EXACTLY:\n"
        "OK <reason> if well-segmented\n"
        "CROSS <reason> if multiple items merged\n"
        "MISSING <reason> if content cut off\n"
        "EMPTY <no meaningful content>\n"
        "Keep reason under 8 words."
    )

    try:
        with httpx.Client(timeout=httpx.Timeout(180.0, connect=30.0)) as client:
            r = client.post(
                f"{LM_STUDIO_HOST}/v1/chat/completions",
                json={
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{img_b64}"
                                    },
                                },
                            ],
                        }
                    ],
                    "max_tokens": 40,
                    "temperature": 0,
                },
            )
            return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        err = str(e)
        if "crashed" in err and attempt < 3:
            print(f"    Model crashed, retrying in 5s (attempt {attempt + 1})...")
            time.sleep(5)
            return verify_zone(pdf_path, page_num, zone, label, attempt=attempt + 1)
        return f"ERROR: {e}"


def llm_analyze_report(report_text):
    import httpx

    prompt = (
        "You analyze a PDF page segmentation report. "
        "Identify suspicious gaps where content may be missing. "
        "Reply with specific zone coordinates to add or adjust.\n\n"
        f"{report_text[:3000]}"
    )
    try:
        with httpx.Client(timeout=httpx.Timeout(60.0, connect=15.0)) as client:
            r = client.post(
                f"{LM_STUDIO_HOST}/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 500,
                    "temperature": 0,
                },
            )
        body = r.json()
        if "choices" in body and len(body["choices"]) > 0:
            return body["choices"][0]["message"]["content"].strip()
        return f"Unexpected response: {str(body)[:500]}"
    except Exception as e:
        return f"ERROR: {e}"


# --- LLM proofreading report ---


def write_llm_report(pages, llm_zones):
    import math

    ph = pages[0]["height"] if pages else 803
    pw = 612

    def fmt_rect(z):
        return f"({z['x0']:.0f},{z['y_bot']:.0f},{z['x1']:.0f},{z['y_top']:.0f})"

    def gap_size(a, b):
        return max(0, a["y_bot"] - b["y_top"]) if a["y_bot"] > b["y_top"] else 0

    lines = []
    lines.append(
        f"**File:** {PDF_FILENAME}  \n"
        f"**Pages:** {len(pages)}  \n"
        f"**Total zones:** {len(llm_zones)}  \n"
        f"**Page size:** {pw} × {ph} pt (Y=0 at bottom)"
    )
    lines.append("")

    for page in pages:
        pn = page["num"]
        pz = sorted(
            [z for z in llm_zones if z["page"] == pn],
            key=lambda z: -z["y_top"],  # top-to-bottom
        )
        ink = page.get("ink_components", [])
        lines.append(f"---")
        lines.append(f"## Page {pn}")
        lines.append(
            f"Zones: {len(pz)} | Ink components: {len([c for c in ink if c['pixels'] >= MIN_DRAWING_PIXELS])}"
        )
        lines.append("")

        # zone table
        lines.append("| # | Label | rect (L,B,R,T) | W | H |")
        lines.append("|---|-------|----------------|---|---|")
        for i, z in enumerate(pz, 1):
            lines.append(
                f"| {i} | {z['label']} | {fmt_rect(z)} | {z['w']:.0f} | {z['h']:.0f} |"
            )
        lines.append("")

        # gaps between zones (vertical)
        sorted_bot = sorted(pz, key=lambda z: z["y_bot"])
        gaps = []
        for i in range(len(sorted_bot) - 1):
            gap = gap_size(sorted_bot[i + 1], sorted_bot[i])
            if gap > 5:
                gaps.append(
                    f"gap {gap:.0f}pt between zone at Y={sorted_bot[i]['y_top']:.0f}–{sorted_bot[i + 1]['y_bot']:.0f}"
                )
        if gaps:
            lines.append("### Vertical gaps between zones")
            for g in gaps:
                lines.append(f"- {g}")
            lines.append("")

        # uncovered page edges
        edges = []
        leftmost = min(z["x0"] for z in pz) if pz else pw
        rightmost = max(z["x1"] for z in pz) if pz else 0
        bottommost = min(z["y_bot"] for z in pz) if pz else ph
        topmost = max(z["y_top"] for z in pz) if pz else 0
        if leftmost > 5:
            edges.append(f"left margin {leftmost:.0f}pt")
        if pw - rightmost > 5:
            edges.append(f"right margin {pw - rightmost:.0f}pt")
        if bottommost > 5:
            edges.append(f"bottom margin {bottommost:.0f}pt")
        if ph - topmost > 5:
            edges.append(f"top margin {ph - topmost:.0f}pt")

        # also check: gaps between zones in X direction (horizontal splits)
        # group zones by Y-band and check horizontal gaps
        bands = {}
        for z in sorted_bot:
            band = round(z["y_bot"] / 40)
            bands.setdefault(band, []).append(z)
        h_gaps = []
        for band, bz in bands.items():
            if len(bz) < 2:
                continue
            sorted_x = sorted(bz, key=lambda z: z["x0"])
            for i in range(len(sorted_x) - 1):
                hgap = sorted_x[i + 1]["x0"] - sorted_x[i]["x1"]
                if 10 < hgap < 300:
                    # only flag if the gap area contains ink
                    gx0 = sorted_x[i]["x1"]
                    gx1 = sorted_x[i + 1]["x0"]
                    gy_bot = max(z["y_bot"] for z in [sorted_x[i], sorted_x[i + 1]])
                    gy_top = min(z["y_top"] for z in [sorted_x[i], sorted_x[i + 1]])
                    for comp in ink:
                        if comp["pixels"] < MIN_DRAWING_PIXELS:
                            continue
                        ox = max(gx0, comp["x0"]) < min(gx1, comp["x1"])
                        oy = max(gy_bot, comp["y_bot"]) < min(gy_top, comp["y_top"])
                        if ox and oy:
                            h_gaps.append(
                                f"horizontal gap {hgap:.0f}pt at Y={gy_bot:.0f}–{gy_top:.0f} (has ink)"
                            )
                            break

        if edges:
            lines.append("### Uncovered page margins")
            for e in edges:
                lines.append(f"- {e}")
            lines.append("")

        if gaps or h_gaps:
            lines.append("### ⚠ Potential issues")
            for g in gaps:
                lines.append(f"- {g}")
            for h in h_gaps:
                lines.append(f"- {h}")
            lines.append("")

    lines.append("---")
    lines.append("## Review instructions")
    lines.append(
        "Check each page for: 1) zones that are too large (swallowing multiple items), "
        "2) zones that are too small (cutting content), "
        "3) vertical gaps with ink not captured by any zone, "
        "4) horizontal gaps between zones that may contain missed strokes."
    )
    lines.append("")

    with open(LLM_OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    issue_count = len(gaps) + len(h_gaps) + len(edges)
    print(
        f"Wrote {LLM_OUTPUT_PATH} ({len(llm_zones)} zones, {issue_count} flagged gaps/margins)"
    )


def main():
    pages = extract_notability_objects(PDF_PATH)
    print(f"Extracted {sum(len(p['objects']) for p in pages)} text objects")

    extract_embedded_images(PDF_PATH, pages)
    find_ink_components(PDF_PATH, pages)
    find_drawing_regions(pages)
    find_box_zones(PDF_PATH, pages)

    print("Running Docling layout analysis...")
    docling_items = get_docling_layout(PDF_PATH)

    # Check LM Studio
    lm_ok = False
    if LM_VERIFY:
        print("Checking LM Studio...")
        if check_lm_studio():
            lm_ok = True
            print("  Will verify zones via LM Studio.")
        else:
            print("  WARNING: LM Studio not running. Skipping VLM verification.")

    lines = []
    llm_zones = []
    total_text = total_drawing = total_missed = total_uncovered = 0

    for page in pages:
        page_items = [i for i in docling_items if i["page"] == page["num"]]

        # --- build text zones ---
        text_zones = []
        for c in y_cluster_pdf(page["objects"]):
            text_zones.append(
                {
                    "x0": min(o["x0"] for o in c),
                    "x1": max(o["x1"] for o in c),
                    "y_bot": min(o["y_bot"] for o in c),
                    "y_top": max(o["y_top"] for o in c),
                    "label": classify(c, page_items),
                }
            )

        # --- build drawing zones ---
        drawing_zones = []
        for drw in page.get("drawings", []):
            w = drw["x1"] - drw["x0"]
            h = drw["y_top"] - drw["y_bot"]
            if w * h < 500:
                continue
            drawing_zones.append({**drw, "label": guess_drawing_label(w, h)})

        # --- build image zones ---
        image_zones = [{**img, "label": "image"} for img in page.get("images", [])]

        # --- suppress text zones swallowed by drawings (except formulas) ---
        keep_text = []
        for tz in text_zones:
            swallowed = False
            for dz in drawing_zones + image_zones:
                if overlap_ratio(tz, dz) > 0.4 and tz["label"] != "formula":
                    if dz in drawing_zones:
                        dz["x0"] = min(dz["x0"], tz["x0"])
                        dz["x1"] = max(dz["x1"], tz["x1"])
                        dz["y_bot"] = min(dz["y_bot"], tz["y_bot"])
                        dz["y_top"] = max(dz["y_top"], tz["y_top"])
                    swallowed = True
                    break
            if not swallowed:
                keep_text.append(tz)

        # --- merge adjacent formulas into "demarche" zones ---
        formulas = [z for z in keep_text if z["label"] == "formula"]
        others = [z for z in keep_text if z["label"] != "formula"]
        merged = []
        if formulas:
            sorted_f = sorted(formulas, key=lambda z: z["y_bot"])
            cur = [sorted_f[0]]
            for f in sorted_f[1:]:
                gap = abs(f["y_bot"] - cur[-1]["y_top"])
                if gap <= 30:
                    cur.append(f)
                else:
                    merged.append(cur)
                    cur = [f]
            merged.append(cur)
        demarche_zones = []
        for group in merged:
            demarche_zones.append(
                {
                    "x0": min(z["x0"] for z in group),
                    "x1": max(z["x1"] for z in group),
                    "y_bot": min(z["y_bot"] for z in group),
                    "y_top": max(z["y_top"] for z in group),
                    "label": "demarche",
                }
            )
        keep_text = others + demarche_zones

        # --- build & apply box zones ---
        box_zones = page.get("boxes", [])
        if box_zones:
            # suppress drawing zones that overlap heavily with boxes (expand box to cover)
            filtered_drawings = []
            for dz in drawing_zones:
                suppressed = False
                for bz in box_zones:
                    if overlap_ratio(dz, bz) > 0.4:
                        bz["x0"] = min(bz["x0"], dz["x0"])
                        bz["x1"] = max(bz["x1"], dz["x1"])
                        bz["y_bot"] = min(bz["y_bot"], dz["y_bot"])
                        bz["y_top"] = max(bz["y_top"], dz["y_top"])
                        suppressed = True
                        break
                if not suppressed:
                    filtered_drawings.append(dz)
            drawing_zones = filtered_drawings
            # suppress text zones inside boxes (expand box to cover them)
            filtered = []
            for tz in keep_text:
                swallowed = False
                for bz in box_zones:
                    if overlap_ratio(tz, bz) > 0.5:
                        bz["x0"] = min(bz["x0"], tz["x0"])
                        bz["x1"] = max(bz["x1"], tz["x1"])
                        bz["y_bot"] = min(bz["y_bot"], tz["y_bot"])
                        bz["y_top"] = max(bz["y_top"], tz["y_top"])
                        swallowed = True
                        break
                if not swallowed:
                    filtered.append(tz)
            keep_text = filtered + box_zones

        # --- missed ink components (large blobs only) ---
        all_zones = keep_text + image_zones + drawing_zones
        missed = []
        for comp in page.get("ink_components", []):
            if comp["pixels"] < MIN_DRAWING_PIXELS:
                continue
            covered = False
            for z in all_zones:
                if overlap_ratio(comp, z) > 0.2:
                    covered = True
                    break
            if not covered:
                missed.append(comp)

        # --- merge & dedup zones ---
        all_zones = merge_overlapping(all_zones)
        all_zones = remove_contained(all_zones)

        # --- find & fill uncovered ink (pixel-level) ---
        uncovered = find_uncovered_ink(PDF_PATH, page["num"], all_zones, page["height"])
        missed_zones = []
        for blob in uncovered:
            if (blob["x1"] - blob["x0"]) * (blob["y_top"] - blob["y_bot"]) < 20:
                continue
            missed_zones.append({**blob, "label": "notability-missed"})
        n_uncovered = len(missed_zones)
        missed_zones = merge_proximate(missed_zones, xy_gap=12)
        all_zones += missed_zones
        all_zones = remove_contained(all_zones)

        # --- VLM verify each zone (skip — text-only model) ---
        verifications = {}
        if lm_ok:
            pass  # vision test skipped; model doesn't support images

        # --- output zones ---
        for z in all_zones:
            rect = f"{z['x0']:.0f},{z['y_bot']:.0f},{z['x1']:.0f},{z['y_top']:.0f}"
            v = verifications.get(id(z), "")
            v_comment = f" -- VLM: {v}" if v else ""
            lines.append(
                f"![[{PDF_FILENAME}#page={page['num']}&rect={rect}|{PDF_FILENAME}, p.{page['num']}]]  <!-- {z['label']}{v_comment} -->"
            )
            llm_zones.append(
                {
                    "page": page["num"],
                    "label": z["label"],
                    "x0": z["x0"],
                    "y_bot": z["y_bot"],
                    "x1": z["x1"],
                    "y_top": z["y_top"],
                    "w": z["x1"] - z["x0"],
                    "h": z["y_top"] - z["y_bot"],
                }
            )

        for m in missed:
            rect = f"{m['x0']:.0f},{m['y_bot']:.0f},{m['x1']:.0f},{m['y_top']:.0f}"
            lines.append(
                f"![[{PDF_FILENAME}#page={page['num']}&rect={rect}|{PDF_FILENAME}, p.{page['num']}]]  <!-- missed-large -->"
            )

        total_text += len(keep_text)
        total_drawing += len(image_zones) + len(drawing_zones)
        total_missed += len(missed)
        total_uncovered = (
            total_uncovered + n_uncovered if "total_uncovered" in dir() else n_uncovered
        )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    parts = [
        f"{len(llm_zones)} zones",
        f"{total_uncovered} uncovered filled",
        f"{total_missed} large missed",
    ]
    print(f"\nWrote {len(lines)} zones to {OUTPUT_PATH} ({', '.join(parts)})")
    write_llm_report(pages, llm_zones)

    if lm_ok:
        print("Running LLM gap analysis on zone report...")
        with open(LLM_OUTPUT_PATH, "r", encoding="utf-8") as f:
            report_text = f.read()
        suggestions = llm_analyze_report(report_text)
        print("\n--- LLM Gap Analysis ---")
        print(suggestions)
        print("------------------------\n")


if __name__ == "__main__":
    main()
