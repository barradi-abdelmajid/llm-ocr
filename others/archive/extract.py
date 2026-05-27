"""PDF data extraction — text objects, images, ink components, drawings, box zones."""

import fitz
import numpy as np
from scipy import ndimage

from config import (
    PDF_PATH,
    MIN_AREA,
    INK_THRESHOLD,
    MIN_DRAWING_PIXELS,
    BOX_MARGIN_PT,
)


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
        pages.append(
            {"num": pn + 1, "objects": objs, "height": ph, "width": page.rect.width}
        )
    doc.close()
    return pages


def extract_embedded_images(pdf_path, pages):
    doc = fitz.open(pdf_path)
    for pn in range(doc.page_count):
        page = doc[pn]
        blocks = page.get_text("dict")["blocks"]
        ph = page.rect.height
        pw = page.rect.width
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

            boxes.append(
                {
                    "x0": max(0, px0 / pix.width * pw - BOX_MARGIN_PT),
                    "x1": min(pw, px1 / pix.width * pw + BOX_MARGIN_PT),
                    "y_bot": max(0, ph - py1 / pix.height * ph - BOX_MARGIN_PT),
                    "y_top": min(ph, ph - py0 / pix.height * ph + BOX_MARGIN_PT),
                    "label": "encadre",
                }
            )
        pages[pn]["boxes"] = boxes
    doc.close()
