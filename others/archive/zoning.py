"""Zone operations — clustering, splitting, merging, filtering, classification."""

from collections import Counter
import numpy as np
from scipy import ndimage

from config import (
    PDF_PATH,
    Y_GAP,
    X_GAP,
    INK_THRESHOLD,
    VBAND_Y_GAP,
    VBAND_X_OVERLAP,
    MERGE_Y_GAP,
    MERGE_X_OVERLAP,
    MERGE_X_GAP,
    MERGE_MAX_H,
    MERGE_H_GAP,
    MERGE_H_Y_OVERLAP,
    DIVIDER_MAX_H,
    DIVIDER_MIN_W_FRAC,
    COLOR_THRESHOLD,
    BLUE_BIAS,
    COLORED_RATIO,
    GUARD_SPLIT_GAP,
    MAX_ZONE_HEIGHT_FRAC,
    TEXT_DRAWING_OVERLAP,
    TEXT_BOX_OVERLAP,
    UNCOVERED_MIN_PIXELS,
    UNCOVERED_MIN_AREA,
    TINY_MIN_AREA,
    TINY_MIN_W,
    TINY_MIN_H,
    MIN_DRAWING_PIXELS,
)

import fitz


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


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


def x_split_cluster(objects, gap=X_GAP):
    if not objects:
        return []
    sorted_objs = sorted(objects, key=lambda o: o["x0"])
    clusters = [[sorted_objs[0]]]
    for obj in sorted_objs[1:]:
        prev = clusters[-1][-1]
        if obj["x0"] - prev["x1"] > gap:
            clusters.append([obj])
        else:
            clusters[-1].append(obj)
    if len(clusters) > 1:
        merged = [clusters[0]]
        for c in clusters[1:]:
            if len(c) < 3 and len(merged[-1]) >= 3:
                merged[-1].extend(c)
            else:
                merged.append(c)
        clusters = merged
    return clusters


# ---------------------------------------------------------------------------
# Docling layout
# ---------------------------------------------------------------------------


def get_docling_layout(pdf_path):
    from docling.document_converter import DocumentConverter

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


# ---------------------------------------------------------------------------
# Overlap / merge helpers
# ---------------------------------------------------------------------------


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


def merge_union(a, b, margin_pt=4):
    return {
        "x0": max(0, min(a["x0"], b["x0"]) - margin_pt),
        "x1": max(a["x1"], b["x1"]) + margin_pt,
        "y_bot": max(0, min(a["y_bot"], b["y_bot"]) - margin_pt),
        "y_top": max(a["y_top"], b["y_top"]) + margin_pt,
        "label": a.get("label", "text"),
    }


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


# ---------------------------------------------------------------------------
# Merge strategies
# ---------------------------------------------------------------------------


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


def merge_proximate(zones, xy_gap=8):
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
                    b["x0"] >= a["x0"] - 2
                    and b["x1"] <= a["x1"] + 2
                    and b["y_bot"] >= a["y_bot"] - 2
                    and b["y_top"] <= a["y_top"] + 2
                ):
                    removed.add(j)
                    changed = True
        kept = [z for idx, z in enumerate(kept) if idx not in removed]
    return kept


def merge_vertical_bands(zones, y_gap=20, x_overlap_ratio=0.25):
    if not zones:
        return []
    sorted_zones = sorted(zones, key=lambda z: z["y_bot"])
    merged = [dict(sorted_zones[0])]
    for z in sorted_zones[1:]:
        prev = merged[-1]
        pw = min(prev["x1"] - prev["x0"], z["x1"] - z["x0"])
        ox0 = max(prev["x0"], z["x0"])
        ox1 = min(prev["x1"], z["x1"])
        overlap_w = max(0, ox1 - ox0)
        vgap = z["y_bot"] - prev["y_top"]
        if pw > 0 and overlap_w / pw > x_overlap_ratio and vgap <= y_gap:
            merged[-1] = merge_union(prev, z, margin_pt=2)
        else:
            merged.append(dict(z))
    return merged


def merge_neighbors(zones, y_gap=16, x_overlap_ratio=0.03, max_h=500, x_gap=12):
    """Iteratively merge vertically adjacent zones that are horizontally close.
    Height-guarded to prevent chain-reaction full-page zones."""
    if not zones:
        return []
    kept = list(zones)
    changed = True
    while changed:
        changed = False
        sorted_zones = sorted(kept, key=lambda z: z["y_bot"])
        new_kept = [dict(sorted_zones[0])]
        for z in sorted_zones[1:]:
            prev = new_kept[-1]
            merged_h = max(z["y_top"], prev["y_top"]) - min(z["y_bot"], prev["y_bot"])
            if merged_h > max_h:
                new_kept.append(dict(z))
                continue
            px0 = prev["x0"] - x_gap
            px1 = prev["x1"] + x_gap
            ox0 = max(px0, z["x0"])
            ox1 = min(px1, z["x1"])
            overlap_w = max(0, ox1 - ox0)
            pw = min(prev["x1"] - prev["x0"], z["x1"] - z["x0"]) + x_gap
            vgap = z["y_bot"] - prev["y_top"]
            if pw > 0 and overlap_w / pw > x_overlap_ratio and vgap <= y_gap:
                new_kept[-1] = merge_union(prev, z, margin_pt=3)
                changed = True
            else:
                new_kept.append(dict(z))
        kept = new_kept
    return kept


def merge_drawing_text(zones, x_gap=MERGE_H_GAP, y_overlap_ratio=MERGE_H_Y_OVERLAP):
    """Merge drawing/circuit zones with horizontally adjacent text zones.
    Only merges non-text zones into text zones (not text-text)."""
    drawing_labels = {"circuit", "drawing", "image", "encadre"}
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
                n = kept[j]
                # at least one must be a drawing/circuit, other must be text-like
                cur_is_drawing = cur.get("label") in drawing_labels
                n_is_drawing = n.get("label") in drawing_labels
                if cur_is_drawing == n_is_drawing:
                    continue
                # vertical overlap check
                oy_bot = max(cur["y_bot"], n["y_bot"])
                oy_top = min(cur["y_top"], n["y_top"])
                if oy_bot >= oy_top:
                    continue
                overlap_h = oy_top - oy_bot
                cur_h = cur["y_top"] - cur["y_bot"]
                n_h = n["y_top"] - n["y_bot"]
                min_h = min(cur_h, n_h)
                if min_h > 0 and overlap_h / min_h < y_overlap_ratio:
                    continue
                gap = (
                    max(0, n["x0"] - cur["x1"])
                    if n["x0"] > cur["x1"]
                    else max(0, cur["x0"] - n["x1"])
                )
                if gap <= x_gap:
                    cur = merge_union(cur, n, margin_pt=3)
                    merged.add(j)
                    changed = True
            new_kept.append(cur)
        kept = new_kept
    return kept


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def filter_divider_lines(zones, page_width):
    kept = []
    for z in zones:
        w = z["x1"] - z["x0"]
        h = z["y_top"] - z["y_bot"]
        if h < DIVIDER_MAX_H and w > page_width * DIVIDER_MIN_W_FRAC and w > h * 5:
            continue
        kept.append(z)
    return kept


def filter_tiny_zones(zones):
    return [
        z
        for z in zones
        if (z["x1"] - z["x0"]) * (z["y_top"] - z["y_bot"]) >= TINY_MIN_AREA
        and (z["x1"] - z["x0"]) >= TINY_MIN_W
        and (z["y_top"] - z["y_bot"]) >= TINY_MIN_H
    ]


def clamp_to_page(zones, pw, ph):
    for z in zones:
        z["x0"] = max(0, z["x0"])
        z["x1"] = min(pw, z["x1"])
        z["y_bot"] = max(0, z["y_bot"])
        z["y_top"] = min(ph, z["y_top"])
    return zones


# ---------------------------------------------------------------------------
# Ink-based operations
# ---------------------------------------------------------------------------


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


def relabel_by_ink_color(pdf_path, pages):
    doc = fitz.open(pdf_path)
    text_labels = {
        "text",
        "notability-missed",
        "section_header",
        "formula",
        "list_item",
    }
    for page in pages:
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

        for z in page.get("zones", []):
            if z.get("label") not in text_labels:
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
            colored_ratio = (colorness > COLOR_THRESHOLD).sum() / n_ink

            if colored_ratio > COLORED_RATIO:
                blue_bias = ink_b.mean() - ink_r.mean()
                if blue_bias > BLUE_BIAS:
                    z["label"] = "formula"
                else:
                    z["label"] = "section_header"

    doc.close()


def guard_split_oversized_zones(text_zones, page_objects, page_items, page_height):
    """Split text zones covering > MAX_ZONE_HEIGHT_FRAC of page at large gaps."""
    max_zone_h = page_height * MAX_ZONE_HEIGHT_FRAC
    safe = []
    for tz in text_zones:
        if tz["y_top"] - tz["y_bot"] > max_zone_h:
            inner = [
                o
                for o in page_objects
                if o["y_bot"] >= tz["y_bot"]
                and o["y_top"] <= tz["y_top"]
                and o["x0"] >= tz["x0"]
                and o["x1"] <= tz["x1"]
            ]
            sorted_objs = sorted(inner, key=lambda o: o["cy_pdf"])
            cur = [sorted_objs[0]]
            for obj in sorted_objs[1:]:
                gap = obj["cy_pdf"] - cur[-1]["cy_pdf"]
                if gap > GUARD_SPLIT_GAP:
                    safe.append(
                        {
                            "x0": min(o["x0"] for o in cur),
                            "x1": max(o["x1"] for o in cur),
                            "y_bot": min(o["y_bot"] for o in cur),
                            "y_top": max(o["y_top"] for o in cur),
                            "label": classify(cur, page_items),
                        }
                    )
                    cur = [obj]
                else:
                    cur.append(obj)
            if cur:
                safe.append(
                    {
                        "x0": min(o["x0"] for o in cur),
                        "x1": max(o["x1"] for o in cur),
                        "y_bot": min(o["y_bot"] for o in cur),
                        "y_top": max(o["y_top"] for o in cur),
                        "label": classify(cur, page_items),
                    }
                )
        else:
            safe.append(tz)
    return safe
