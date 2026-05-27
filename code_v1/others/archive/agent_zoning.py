"""
Agent-based zoning: LLM uses tools to iteratively refine PDF zoning.
Phase 1: Full pipeline (Notability OCR, images, ink components, boxes, drawings)
Phase 2: Auto-fix obvious issues (cover uncovered ink, merge proximate)
Phase 3: LLM text-based analysis + suggestions
Phase 4: Output
"""

import sys, io, base64, json, time, re

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


# ----- helpers (shared) -----


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


def merge_union(a, b, margin_pt=6, clamp_w=None, clamp_h=None):
    return {
        "x0": max(0, min(a["x0"], b["x0"]) - margin_pt),
        "x1": min(clamp_w or 9999, max(a["x1"], b["x1"]) + margin_pt),
        "y_bot": max(0, min(a["y_bot"], b["y_bot"]) - margin_pt),
        "y_top": min(clamp_h or 9999, max(a["y_top"], b["y_top"]) + margin_pt),
        "label": a.get("label", "text"),
    }


def zones_overlap_bbox(a, b):
    return (
        a["x0"] < b["x1"]
        and a["x1"] > b["x0"]
        and a["y_bot"] < b["y_top"]
        and a["y_top"] > b["y_bot"]
    )


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


def remove_contained(zones):
    """Remove zones that are fully inside another zone (keep the outer)."""
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
                # Check if b is inside a
                if (
                    b["x0"] >= a["x0"] - 2
                    and b["x1"] <= a["x1"] + 2
                    and b["y_bot"] >= a["y_bot"] - 2
                    and b["y_top"] <= a["y_top"] + 2
                ):
                    removed.add(j)
                    changed = True
        kept = [z for idx, z in enumerate(kept) if idx not in removed]
    return kept


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


# ----- Phase 1: Full pipeline extraction (from main.py) -----


def extract_notability_objects(pdf_path, pages):
    doc = fitz.open(pdf_path)
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
        pages[pn]["objects"] = objs
    doc.close()


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
                z = {
                    "x0": comp["x0"],
                    "x1": comp["x1"],
                    "y_bot": comp["y_bot"],
                    "y_top": comp["y_top"],
                }
                w = z["x1"] - z["x0"]
                h = z["y_top"] - z["y_bot"]
                if w * h >= 500:
                    z["label"] = guess_drawing_label(w, h)
                    zones.append(z)
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
            if bw < bh * 0.3 or bh < bw * 0.3:
                continue
            hole_area = len(ys)
            bbox_area = bw * bh
            if hole_area / bbox_area < 0.5:
                continue
            margin = max(2, min(bw, bh) // 20)
            ox0, oy0 = max(0, px0 - margin), max(0, py0 - margin)
            ox1, oy1 = min(w - 1, px1 + margin), min(h - 1, py1 + margin)
            top_strip = ink[oy0 : py0 + 1, ox0 : ox1 + 1]
            bot_strip = ink[py1 : oy1 + 1, ox0 : ox1 + 1]
            lft_strip = ink[oy0 : oy1 + 1, ox0 : px0 + 1]
            rgt_strip = ink[oy0 : oy1 + 1, px1 : ox1 + 1]
            min_ink = max(8, margin * 2)
            if any(
                s.sum() < min_ink for s in [top_strip, bot_strip, lft_strip, rgt_strip]
            ):
                continue
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


def find_uncovered_ink(pdf_path, page_num, zones, page_height, min_pixels=15):
    doc = fitz.open(pdf_path)
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


def extract_phase1(pdf_path):
    """Full Phase 1 extraction."""
    doc = fitz.open(pdf_path)
    pages = []
    for pn in range(doc.page_count):
        page = doc[pn]
        pages.append(
            {
                "num": pn + 1,
                "height": page.rect.height,
                "width": page.rect.width,
                "objects": [],
                "images": [],
                "ink_components": [],
                "drawings": [],
                "boxes": [],
                "zones": [],
            }
        )
    doc.close()

    print("  Extracting Notability text objects...")
    extract_notability_objects(pdf_path, pages)
    print("  Extracting embedded images...")
    extract_embedded_images(pdf_path, pages)
    print("  Finding ink components...")
    find_ink_components(pdf_path, pages)
    print("  Finding drawing regions...")
    find_drawing_regions(pages)
    print("  Finding box zones...")
    find_box_zones(pdf_path, pages)
    print("  Docling layout analysis...")
    docling_items = get_docling_layout(pdf_path)

    for page in pages:
        pn = page["num"]
        page_items = [i for i in docling_items if i["page"] == pn]

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

        drawing_zones = [dict(d) for d in page.get("drawings", [])]
        image_zones = [{**img, "label": "image"} for img in page.get("images", [])]

        # Suppress text swallowed by drawings (except formulas)
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

        # Merge adjacent formulas into "demarche"
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

        # Apply box zones
        box_zones = page.get("boxes", [])
        if box_zones:
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

        all_zones = keep_text + image_zones + drawing_zones
        all_zones = merge_overlapping(all_zones)
        all_zones = remove_contained(all_zones)

        # Pixel-level uncovered ink
        uncovered = find_uncovered_ink(pdf_path, page["num"], all_zones, page["height"])
        missed_zones = []
        for blob in uncovered:
            if (blob["x1"] - blob["x0"]) * (blob["y_top"] - blob["y_bot"]) < 20:
                continue
            missed_zones.append({**blob, "label": "notability-missed"})
        missed_zones = merge_proximate(missed_zones, xy_gap=12)
        all_zones += missed_zones
        all_zones = remove_contained(all_zones)

        for z in all_zones:
            if "source" not in z:
                z["source"] = "phase1"
        for z in missed_zones:
            z["source"] = "uncovered-ink"

        page["zones"] = all_zones

    return pages


# ----- Phase 2: Auto-fix (obvious issues, no LLM needed) -----


def auto_cover_ink(agent):
    """On each page, auto-add zones for uncovered ink."""
    total = 0
    for page in agent.pages:
        blobs = agent.find_uncovered_ink(page["num"])
        if not blobs:
            continue
        merged = merge_proximate(
            [{**b, "label": "text", "source": "auto_covered"} for b in blobs],
            xy_gap=12,
        )
        pw, ph = page["width"], page["height"]
        for z in merged:
            z["id"] = agent.next_id
            agent.next_id += 1
            z["x0"] = max(0, z["x0"])
            z["x1"] = min(pw, z["x1"])
            z["y_bot"] = max(0, z["y_bot"])
            z["y_top"] = min(ph, z["y_top"])
            page["zones"].append(z)
        total += len(merged)
    return total


def auto_merge_nearby(agent, xy_gap=10, max_height=250):
    """Merge zones that are very close together (same label, same page).
    Won't merge if result would exceed max_height."""
    total_merges = 0
    for page in agent.pages:
        changed = True
        pw, ph = page["width"], page["height"]
        while changed:
            changed = False
            zones = sorted(page["zones"], key=lambda z: (z["label"], z["y_top"]))
            i = 0
            while i < len(zones):
                j = i + 1
                merged_any = False
                while j < len(zones):
                    a, b = zones[i], zones[j]
                    if a["label"] != b["label"]:
                        j += 1
                        continue
                    dx = max(0, max(a["x0"], b["x0"]) - min(a["x1"], b["x1"]))
                    dy = max(
                        0, max(a["y_bot"], b["y_bot"]) - min(a["y_top"], b["y_top"])
                    )
                    # Don't merge if zones are far apart or result would be too tall
                    merged_h = max(a["y_top"], b["y_top"]) - min(a["y_bot"], b["y_bot"])
                    if dy > 30 or merged_h > max_height:
                        j += 1
                        continue
                    if dx < xy_gap and dy < xy_gap and dx + dy < xy_gap * 2:
                        merged = merge_union(a, b, margin_pt=4, clamp_w=pw, clamp_h=ph)
                        merged["id"] = agent.next_id
                        agent.next_id += 1
                        merged["source"] = "auto_merged"
                        zones.pop(j)
                        zones[i] = merged
                        merged_any = True
                        total_merges += 1
                        changed = True
                    else:
                        j += 1
                if not merged_any:
                    i += 1
            page["zones"] = zones
    return total_merges


# ----- Phase 3: LLM analysis (text only) -----


def build_text_report(agent):
    """Build a condensed text report for LLM analysis."""
    lines = [f"PDF: {PDF_FILENAME}, {len(agent.pages)} pages"]
    for page in agent.pages:
        zones = sorted(page["zones"], key=lambda z: z["y_top"])
        lines.append(f"\n## Page {page['num']} ({len(zones)} zones)")
        lines.append("ID | Label | Source | Rect L,B,R,T | WxH")
        for z in zones:
            r = f"{z['x0']:.0f},{z['y_bot']:.0f},{z['x1']:.0f},{z['y_top']:.0f}"
            w = z["x1"] - z["x0"]
            h = z["y_top"] - z["y_bot"]
            lines.append(
                f"{z['id']} | {z['label']} | {z.get('source', '')} | {r} | {w:.0f}x{h:.0f}"
            )
        # Uncovered ink
        blobs = agent.find_uncovered_ink(page["num"])
        if blobs:
            lines.append(f"UNCOVERED: {len(blobs)} ink blobs outside all zones")
        else:
            lines.append("UNCOVERED: none")
    return "\n".join(lines)


def llm_analyze_report(report_text):
    import httpx

    prompt = (
        "You analyze a PDF page zone segmentation of handwritten electrical engineering notes.\n"
        "Report issues you find. Be specific: mention zone IDs and what to do.\n"
        "Possible issues:\n"
        "- Zones that should be merged (same content, very close)\n"
        "- Zones that are too small (cutting content)\n"
        "- Zones that are too large (swallowing unrelated content)\n"
        "- Zones incorrectly labeled\n"
        "- Gaps between zones that likely contain ink (mention approximate coordinates)\n"
        "- Uncovered ink that should be captured\n\n"
        "Reply with specific, actionable suggestions. Format each as:\n"
        "ACTION: <zone_id(s)> <merge|resize|delete|relabel> <reason>\n"
        "Or: GAP: page N at (x0,y0,x1,y1) likely has ink\n\n"
        f"{report_text}"
    )
    try:
        with httpx.Client(timeout=httpx.Timeout(120.0, connect=15.0)) as client:
            r = client.post(
                f"{LM_STUDIO_HOST}/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 1000,
                    "temperature": 0,
                },
            )
        body = r.json()
        if "choices" in body and len(body["choices"]) > 0:
            return body["choices"][0]["message"]["content"].strip()
        return f"Unexpected response: {str(body)[:500]}"
    except Exception as e:
        return f"ERROR: {e}"


def _parse_action_line(line):
    """Extract action type and zone IDs from various LLM formats.
    Returns (action_type, [zone_ids]) or None."""
    # Format 1: **MERGE**: `id` and `id`
    m = re.match(r"(MERGE|DELETE|RESIZE|GAP):\s*", line, re.IGNORECASE)
    if m:
        return m.group(1).upper(), re.findall(r"\d+", line)
    # Format 2: *ACTION: Resize/Delete/Merge ID ...
    m = re.match(r"ACTION:\s*(Resize|Delete|Merge|Merge)\s+", line, re.IGNORECASE)
    if m:
        action_map = {
            "Resize": "RESIZE",
            "Delete": "DELETE",
            "Merge": "MERGE",
            "merge": "MERGE",
        }
        verb = m.group(1)
        return action_map.get(verb, verb.upper()), re.findall(r"\d+", line)
    return None


def apply_llm_suggestions(agent, suggestions):
    actions = []
    for raw_line in suggestions.split("\n"):
        line = re.sub(r"\*", "", raw_line).strip()
        line = re.sub(r"^[\s\-]*\d*\.?\s*", "", line).strip()
        if not line:
            continue

        parsed = _parse_action_line(line)
        if not parsed:
            # Check for GAP/Action patterns that don't start with the keyword
            if re.search(r"\bGAP\b", line, re.IGNORECASE):
                parsed = ("GAP", re.findall(r"\d+", line))
            elif re.search(
                r"\bACTION\b.*\b(Merge|Resize|Delete)\b", line, re.IGNORECASE
            ):
                verb_m = re.search(r"\b(Merge|Resize|Delete)\b", line, re.IGNORECASE)
                if verb_m:
                    parsed = (verb_m.group(1).upper(), re.findall(r"\d+", line))
        if not parsed:
            continue

        action_type, ids = parsed
        ids = [int(x) for x in ids]

        if action_type == "MERGE" and len(ids) >= 2:
            pages_set = set()
            valid = []
            has_image = False
            for zid in ids:
                p, z = agent._find_zone(zid)
                if z:
                    pages_set.add(p["num"])
                    valid.append(zid)
                    if z.get("label") == "image":
                        has_image = True
            if has_image:
                image_ids = [
                    zid
                    for zid in valid
                    if agent._find_zone(zid)[1].get("label") == "image"
                ]
                non_image_ids = [
                    zid
                    for zid in valid
                    if agent._find_zone(zid)[1].get("label") != "image"
                ]
                if image_ids and non_image_ids:
                    actions.append(
                        f"Skipped merge {valid}: image zones should not merge with non-image zones"
                    )
                    continue
            if len(pages_set) == 1 and len(valid) >= 2:
                result = agent.tool_merge(valid)
                actions.append(f"Merged {valid}: {result}")

        elif action_type == "DELETE" and ids:
            zid = ids[0]
            result = agent.tool_delete(zid)
            actions.append(f"Deleted {zid}: {result}")

        elif action_type == "RESIZE" and ids:
            if len(ids) >= 2:
                target_id, other_id = ids[0], ids[1]
                page_t, z_t = agent._find_zone(target_id)
                page_o, z_o = agent._find_zone(other_id)
                if z_t and z_o and page_t["num"] == page_o["num"]:
                    z_t["x0"] = min(z_t["x0"], z_o["x0"])
                    z_t["x1"] = max(z_t["x1"], z_o["x1"])
                    z_t["y_bot"] = min(z_t["y_bot"], z_o["y_bot"])
                    z_t["y_top"] = max(z_t["y_top"], z_o["y_top"])
                    actions.append(f"Resized {target_id} to include {other_id}")
            elif ids:
                actions.append(
                    f"RESIZE suggestion for zone {ids[0]} needs manual review"
                )

        elif action_type == "GAP":
            pairs = re.findall(r"\((\d+\.?\d*)\s*,\s*(\d+\.?\d*)\)", line)
            if pairs:
                pn_match = re.search(r"page\s*(\d+)", line, re.IGNORECASE)
                pn = int(pn_match.group(1)) if pn_match else 1
                if len(pairs) >= 2:
                    x0, y0 = map(float, pairs[0])
                    x1, y1 = map(float, pairs[1])
                else:
                    x0, y0 = map(float, pairs[0])
                    x1, y1 = x0 + 50, y0 + 50
                result = agent.tool_add(pn, x0, y0, x1, y1, "text")
                actions.append(f"Added gap zone: {result}")
    return actions


# ----- Post-processing filters -----


def filter_divider_lines(zones, page_width):
    """Remove zones that are just horizontal divider lines (thin, wide)."""
    kept = []
    for z in zones:
        w = z["x1"] - z["x0"]
        h = z["y_top"] - z["y_bot"]
        # Horizontal line: very thin, spans most of page width
        if h < 8 and w > page_width * 0.4 and w > h * 5:
            continue
        kept.append(z)
    return kept


def cap_zone_height(zones, max_h=250):
    """Don't let any zone exceed max_h vertically."""
    kept = []
    for z in zones:
        h = z["y_top"] - z["y_bot"]
        if h > max_h:
            # Split into chunks of max_h
            n_chunks = int(-(-h // max_h))  # ceil division
            chunk_h = h / n_chunks
            for i in range(n_chunks):
                kept.append(
                    {
                        **z,
                        "y_bot": z["y_bot"] + i * chunk_h,
                        "y_top": min(z["y_top"], z["y_bot"] + (i + 1) * chunk_h),
                        "source": z.get("source", "phase1") + "_split",
                    }
                )
        else:
            kept.append(z)
    return kept


def relabel_by_ink_color(pdf_path, agent):
    """Check ink color in text-like zones and relabel: colored ink → section_header or formula.
    Skips non-text zone types (image, circuit, drawing, encadre, code, demarche)."""
    doc = fitz.open(pdf_path)
    text_labels = {
        "text",
        "notability-missed",
        "uncovered-ink",
        "section_header",
        "formula",
        "list_item",
    }
    for page in agent.pages:
        p = doc[page["num"] - 1]
        pix = p.get_pixmap(dpi=72)
        pw, ph = page["width"], page["height"]
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        if pix.n < 3:
            continue

        gray_full = img[..., :3].mean(axis=2)
        r_full, g_full, b_full = img[..., 0], img[..., 1], img[..., 2]

        for z in page["zones"]:
            if z["label"] not in text_labels:
                continue

            x0 = max(0, int(z["x0"] / pw * pix.width))
            x1 = min(pix.width, int(z["x1"] / pw * pix.width) + 1)
            y0 = max(0, int((ph - z["y_top"]) / ph * pix.height))
            y1 = min(pix.height, int((ph - z["y_bot"]) / ph * pix.height) + 1)

            gray = gray_full[y0:y1, x0:x1]
            r, g, b = r_full[y0:y1, x0:x1], g_full[y0:y1, x0:x1], b_full[y0:y1, x0:x1]

            ink_mask = gray < INK_THRESHOLD
            n_ink = ink_mask.sum()
            if n_ink < 20:
                continue

            ink_r, ink_g, ink_b = r[ink_mask], g[ink_mask], b[ink_mask]
            colorness = np.abs(
                np.maximum(np.maximum(ink_r, ink_g), ink_b)
                - np.minimum(np.minimum(ink_r, ink_g), ink_b)
            )
            colored_ratio = (colorness > 40).sum() / n_ink

            if colored_ratio > 0.25:
                blue_bias = ink_b.mean() - ink_r.mean()
                if blue_bias > 20:
                    z["label"] = "formula"
                else:
                    z["label"] = "section_header"

    doc.close()


# ----- Phase 4: Output -----


def write_output(agent):
    """Write output.md and llm_output.md."""
    lines = []
    llm_zones = []

    for page in agent.pages:
        zones = sorted(page["zones"], key=lambda z: z["y_top"])
        for z in zones:
            rect = f"{z['x0']:.0f},{z['y_bot']:.0f},{z['x1']:.0f},{z['y_top']:.0f}"
            lines.append(
                f"![[{PDF_FILENAME}#page={page['num']}&rect={rect}|{PDF_FILENAME}, p.{page['num']}]]  <!-- {z['label']} ({z.get('source', 'unknown')}) -->"
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
                    "source": z.get("source", "unknown"),
                }
            )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # Write LLM report (final)
    report_lines = [
        f"# Agent Zoning Report\n**File:** {PDF_FILENAME}\n**Pages:** {len(agent.pages)}\n**Total zones:** {len(llm_zones)}\n"
    ]
    for page in agent.pages:
        pz = [z for z in llm_zones if z["page"] == page["num"]]
        report_lines.append(f"\n## Page {page['num']}\nZones: {len(pz)}")
        report_lines.append("| ID | Label | Source | Rect | W | H |")
        report_lines.append("|---|-------|--------|------|---|---|")
        for i, z in enumerate(sorted(pz, key=lambda x: x["y_top"]), 1):
            rect = f"({z['x0']:.0f},{z['y_bot']:.0f},{z['x1']:.0f},{z['y_top']:.0f})"
            report_lines.append(
                f"| {i} | {z['label']} | {z['source']} | {rect} | {z['w']:.0f} | {z['h']:.0f} |"
            )
        # Final uncovered check
        blobs = agent.find_uncovered_ink(page["num"])
        if blobs:
            report_lines.append(f"  **{len(blobs)} uncovered ink blobs remain**")
        else:
            report_lines.append(f"  All ink covered \u2713")

    with open(LLM_OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    return llm_zones, len(llm_zones)


# ----- Main -----


class ZoningAgent:
    def __init__(self, pdf_path, pages):
        self.pdf_path = pdf_path
        self.pages = pages
        self.next_id = 1
        self.doc = fitz.open(pdf_path)
        for page in pages:
            for z in page["zones"]:
                z["id"] = self.next_id
                self.next_id += 1

    def _id(self):
        i = self.next_id
        self.next_id += 1
        return i

    def _find_zone(self, zone_id):
        for page in self.pages:
            for z in page["zones"]:
                if z["id"] == zone_id:
                    return page, z
        return None, None

    def find_uncovered_ink(self, page_num, min_pixels=15):
        page = self.pages[page_num - 1]
        ph = page["height"]
        pw = page["width"]
        p = self.doc[page_num - 1]
        pix = p.get_pixmap(dpi=72)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        gray = img[..., :3].mean(axis=2) if pix.n >= 3 else img[..., 0]
        ink = gray < INK_THRESHOLD
        covered = np.zeros_like(ink, dtype=bool)
        for z in page["zones"]:
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
            area = (
                (xs.max() - xs.min())
                * (ys.max() - ys.min())
                / (pix.width * pix.height)
                * pw
                * ph
            )
            if area < 20:
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
        return blobs

    def tool_merge(self, zone_ids):
        all_zones = []
        for zid in zone_ids:
            page, z = self._find_zone(zid)
            if z is None:
                return f"Zone {zid} not found"
            all_zones.append((page, z))
        if not all_zones:
            return "No valid zones to merge"
        target_page = all_zones[0][0]
        for p, _ in all_zones:
            if p["num"] != target_page["num"]:
                return "Cannot merge zones from different pages"
        merged = all_zones[0][1]
        pw, ph = target_page["width"], target_page["height"]
        for _, z in all_zones[1:]:
            merged = merge_union(merged, z, margin_pt=4, clamp_w=pw, clamp_h=ph)
        merged["id"] = self._id()
        merged["source"] = "llm_merge"
        merged["label"] = all_zones[0][1]["label"]
        ids_to_remove = set(zone_ids)
        target_page["zones"] = [
            z for z in target_page["zones"] if z["id"] not in ids_to_remove
        ]
        target_page["zones"].append(merged)
        return f"Merged zones {zone_ids} into new zone {merged['id']}"

    def tool_resize(self, zone_id, x0, y_bot, x1, y_top):
        page, z = self._find_zone(zone_id)
        if z is None:
            return f"Zone {zone_id} not found"
        z["x0"] = max(0, x0)
        z["x1"] = min(page["width"], x1)
        z["y_bot"] = max(0, y_bot)
        z["y_top"] = min(page["height"], y_top)
        return f"Zone {zone_id} resized to ({x0:.0f},{y_bot:.0f},{x1:.0f},{y_top:.0f})"

    def tool_add(self, page_num, x0, y_bot, x1, y_top, label):
        page = self.pages[page_num - 1]
        z = {
            "id": self._id(),
            "x0": max(0, x0),
            "x1": min(page["width"], x1),
            "y_bot": max(0, y_bot),
            "y_top": min(page["height"], y_top),
            "label": label,
            "source": "llm_added",
        }
        page["zones"].append(z)
        return f"Added zone {z['id']}: {label} at ({x0:.0f},{y_bot:.0f},{x1:.0f},{y_top:.0f})"

    def tool_delete(self, zone_id):
        for page in self.pages:
            before = len(page["zones"])
            page["zones"] = [z for z in page["zones"] if z["id"] != zone_id]
            if len(page["zones"]) < before:
                return f"Deleted zone {zone_id}"
        return f"Zone {zone_id} not found"

    def tool_cover_gaps(self, page_num):
        page = self.pages[page_num - 1]
        blobs = self.find_uncovered_ink(page_num)
        added = []
        for blob in blobs:
            z = {
                "id": self._id(),
                **blob,
                "label": "text",
                "source": "llm_covered",
            }
            page["zones"].append(z)
            added.append(z["id"])
        return (
            f"Added {len(added)} zones for uncovered ink: {added}"
            if added
            else "No uncovered ink found"
        )


def main():
    print("=" * 50)
    print("PDF Zone Segmentation Pipeline")
    print("=" * 50)

    # Phase 1: Full extraction
    print("\n[Phase 1] Full pipeline extraction...")
    pages = extract_phase1(PDF_PATH)
    total = sum(len(p["zones"]) for p in pages)
    print(f"  -> {total} initial zones across {len(pages)} pages")

    agent = ZoningAgent(PDF_PATH, pages)

    # Phase 2: Auto-fix
    print("\n[Phase 2] Auto-fixing obvious issues...")

    n_covered = auto_cover_ink(agent)
    print(f"  Auto-covered {n_covered} uncovered ink zones")

    n_merged = auto_merge_nearby(agent, xy_gap=10)
    print(f"  Auto-merged {n_merged} nearby zones")

    total_after_auto = sum(len(p["zones"]) for p in agent.pages)
    print(f"  -> {total_after_auto} zones after auto-fix")

    # Phase 3: LLM analysis
    print("\n[Phase 3] LLM gap analysis...")
    report = build_text_report(agent)
    print(f"  Report length: {len(report)} chars")
    suggestions = llm_analyze_report(report)
    print(f"\n  --- LLM Suggestions ---")
    print(suggestions)
    print(f"  -----------------------\n")

    # Apply LLM suggestions
    actions = apply_llm_suggestions(agent, suggestions)
    if actions:
        print(f"  Applied {len(actions)} LLM suggestions:")
        for a in actions:
            print(f"    - {a}")
    else:
        print("  No actionable suggestions parsed (text-only review).")

    # Remove contained zones after all changes
    for page in agent.pages:
        page["zones"] = remove_contained(page["zones"])

    # Final auto-cover pass: cover any ink left uncovered by merges/deletes
    print("\n[Phase 3b] Final auto-cover pass...")
    n_final = auto_cover_ink(agent)
    if n_final:
        print(f"  Re-covered {n_final} ink blobs exposed by LLM changes")
    # Remove contained again
    for page in agent.pages:
        page["zones"] = remove_contained(page["zones"])
        # Remove tiny zones (area < 20 pt²)
        page["zones"] = [
            z
            for z in page["zones"]
            if (z["x1"] - z["x0"]) * (z["y_top"] - z["y_bot"]) >= 20
            and (z["x1"] - z["x0"]) >= 3
            and (z["y_top"] - z["y_bot"]) >= 3
        ]

    # Post-processing: cap zone height, filter divider lines, clamp negative coords
    print("\n[Post-process] Cleaning up zones...")
    n_before = sum(len(p["zones"]) for p in agent.pages)
    for page in agent.pages:
        pw, ph = page["width"], page["height"]
        for z in page["zones"]:
            z["x0"] = max(0, z["x0"])
            z["x1"] = min(pw, z["x1"])
            z["y_bot"] = max(0, z["y_bot"])
            z["y_top"] = min(ph, z["y_top"])
        page["zones"] = cap_zone_height(page["zones"], max_h=300)
        page["zones"] = filter_divider_lines(page["zones"], pw)
        # Remove zero-area zones after capping
        page["zones"] = [
            z for z in page["zones"] if z["x1"] > z["x0"] and z["y_top"] > z["y_bot"]
        ]
    n_after = sum(len(p["zones"]) for p in agent.pages)
    print(f"  Cap height + filter dividers: {n_before} -> {n_after} zones")

    # Post-processing: relabel zones by ink color
    print("  Checking ink color for label refinement...")
    relabel_by_ink_color(PDF_PATH, agent)

    # Phase 4: Output
    print("\n[Phase 4] Writing output...")
    zones_list, n_zones = write_output(agent)
    print(f"  -> {n_zones} zones written to {OUTPUT_PATH} and {LLM_OUTPUT_PATH}")
    print("\nDone.")


if __name__ == "__main__":
    main()
