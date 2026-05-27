"""
Visual zone overlay debug tool.
Generates PNG images with zone rectangles overlaid on each PDF page.
Open the output PNGs in Obsidian or any image viewer to review zoning.
"""

import sys, io, os, re, ast

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import fitz
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_config(path="config.txt"):
    cfg = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.split("#")[0].strip()
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip()
            try:
                cfg[k] = ast.literal_eval(v)
            except (ValueError, SyntaxError):
                cfg[k] = v
    return cfg


_cfg = load_config()
PDF_PATH = _cfg.get("PDF_PATH", "pdf_idk.pdf")
OUT_DIR = "debug_zones"

COLORS = {
    "text": "blue",
    "section_header": "red",
    "formula": "lime",
    "image": "orange",
    "circuit": "purple",
    "drawing": "teal",
    "encadre": "magenta",
    "code": "gray",
    "demarche": "royalblue",
    "list_item": "cyan",
    "notability-missed": "yellow",
    "paragraph": "blue",
}


def load_zones_from_md(md_path):
    zones = []
    current_page = None
    with open(md_path, "r", encoding="utf-8") as f:
        for line in f:
            pm = re.match(r"## Page (\d+)", line)
            if pm:
                current_page = int(pm.group(1))
            zm = re.match(
                r"\| \d+ \| ([\w-]+) \| \(([-\d,.]+)\) \| \d+ \| \d+ \|", line
            )
            if zm and current_page:
                label, rect_str = zm.groups()
                coords = [float(x) for x in rect_str.split(",")]
                if len(coords) == 4:
                    zones.append(
                        {
                            "page": current_page,
                            "label": label,
                            "x0": coords[0],
                            "y_bot": coords[1],
                            "x1": coords[2],
                            "y_top": coords[3],
                        }
                    )
    return zones


def draw_zones(pdf_path, zones, out_dir):
    doc = fitz.open(pdf_path)
    os.makedirs(out_dir, exist_ok=True)

    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for pn in range(doc.page_count):
        page = doc[pn]
        pw, ph = page.rect.width, page.rect.height
        page_zones = [z for z in zones if z["page"] == pn + 1]
        if not page_zones:
            continue

        pix = page.get_pixmap(dpi=150)
        scale = pix.width / pw

        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        draw = ImageDraw.Draw(img, "RGBA")

        for z in page_zones:
            x0 = int(z["x0"] * scale)
            x1 = int(z["x1"] * scale)
            y0 = int((ph - z["y_top"]) * scale)
            y1 = int((ph - z["y_bot"]) * scale)
            y0, y1 = min(y0, y1), max(y0, y1)

            color = COLORS.get(z["label"], "gray")
            try:
                r, g, b = ImageColor.getrgb(color)
            except Exception:
                r, g, b = 200, 200, 200

            # Semi-transparent fill + border
            draw.rectangle([x0, y0, x1, y1], outline=(r, g, b, 255), width=3)
            draw.rectangle([x0, y0, x1, y1], fill=(r, g, b, 40))

            # Label
            label = f"{z['label']}"
            bbox = draw.textbbox((0, 0), label, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            label_y = max(0, y0 - th - 4)
            draw.rectangle(
                [x0, label_y, x0 + tw + 4, label_y + th + 4], fill=(0, 0, 0, 200)
            )
            draw.text((x0 + 2, label_y + 2), label, font=font, fill=(r, g, b, 255))

        img.save(os.path.join(out_dir, f"page_{pn + 1}_zones.png"))
        print(
            f"  page {pn + 1}: {len(page_zones)} zones -> {OUT_DIR}/page_{pn + 1}_zones.png"
        )

    doc.close()


if __name__ == "__main__":
    from PIL import ImageColor

    print("Loading zones from llm_output.md...")
    zones = load_zones_from_md("llm_output.md")
    print(f"  {len(zones)} zones loaded")
    draw_zones(PDF_PATH, zones, OUT_DIR)
    print(f"\nDone. Open debug_zones/ folder to see zone overlays.")
    print("Each zone has a colored border + semi-transparent fill + label.")
