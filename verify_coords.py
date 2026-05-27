#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple coordinate verification script for debugging Phase 2 text positioning.
"""

import json
import fitz
from PIL import Image, ImageDraw, ImageFont
import io


def verify_zone_coordinates():
    """Verify coordinate conversions for a sample zone."""

    # Load phase1 data
    with open("phases/phase1_zones.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    pdf_path = data["pdf_path"]
    pdf_dims = data["pdf_dims"]
    zones = data["zones"]

    print(f"PDF: {pdf_path}")
    print(f"Number of pages: {len(pdf_dims)}")
    print(f"Number of zones: {len(zones)}")

    # Open PDF and get first page image (200 DPI)
    doc = fitz.open(pdf_path)
    if doc.page_count == 0:
        raise ValueError("PDF has no pages")
    page = doc[0]  # First page
    if page is None:
        raise ValueError("Failed to get page 0 from PDF")
    pix = page.get_pixmap(dpi=200)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    pdf_page_width = page.rect.width
    pdf_page_height = page.rect.height
    doc.close()

    print(f"Image size: {img.size}")
    print(f"PDF page size: {pdf_page_width} x {pdf_page_height}")

    # Find first non-skip zone with content (we'll check phase2 data)
    try:
        with open("phases/phase2_zones.json", "r", encoding="utf-8") as f:
            phase2_data = json.load(f)
        phase2_zones = phase2_data["zones"]
    except:
        phase2_zones = zones

    # Find a good test zone (not skip, has reasonable size)
    test_zone = None
    for zone in phase2_zones:
        if zone.get("task_type") != "skip":
            bbox = zone["bbox_norm"]
            px1 = int(bbox[0] * img.size[0] / 1000)
            py1 = int(bbox[1] * img.size[1] / 1000)
            px2 = int(bbox[2] * img.size[0] / 1000)
            py2 = int(bbox[3] * img.size[1] / 1000)
            zone_w = px2 - px1
            zone_h = py2 - py1
            if zone_w > 10 and zone_h > 10:  # Reasonable size
                test_zone = zone
                break

    if test_zone is None:
        print("No suitable test zone found")
        return

    print(f"\nTest zone: {test_zone['label']} (page {test_zone['page']})")
    print(f"Task type: {test_zone.get('task_type', 'unknown')}")
    print(f"Content: {repr(test_zone.get('content', ''))[:50]}")

    bbox = test_zone["bbox_norm"]
    px1 = int(bbox[0] * img.size[0] / 1000)
    py1 = int(bbox[1] * img.size[1] / 1000)
    px2 = int(bbox[2] * img.size[0] / 1000)
    py2 = int(bbox[3] * img.size[1] / 1000)

    print(f"Bbox norm: {bbox}")
    print(f"Pixel coords: ({px1}, {py1}) to ({px2}, {py2})")
    print(f"Zone size: {px2 - px1} x {py2 - py1} pixels")

    # Convert to PDF coordinates
    pdf_x1 = px1 * pdf_dims[0]["width"] / img.size[0]
    pdf_x2 = px2 * pdf_dims[0]["width"] / img.size[0]
    pdf_y1 = (img.size[1] - py2) * pdf_dims[0]["height"] / img.size[1]
    pdf_y2 = (img.size[1] - py1) * pdf_dims[0]["height"] / img.size[1]

    print(f"PDF coords: ({pdf_x1:.1f}, {pdf_y1:.1f}) to ({pdf_x2:.1f}, {pdf_y2:.1f})")
    print(f"Zone size in PDF points: {pdf_x2 - pdf_x1:.1f} x {pdf_y2 - pdf_y1:.1f}")

    # Test text rendering
    text = test_zone.get("content", "").strip()
    if not text:
        text = "Test"
    text = text.replace("\n", " ")

    # Calculate font size
    zone_w_px = px2 - px1
    zone_h_px = py2 - py1

    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 100)
    except:
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    w_at_100 = draw.textlength(text, font=font)
    if w_at_100 <= 0:
        w_at_100 = len(text) * 55
    fontsize = zone_w_px / (w_at_100 / 100)
    fontsize = min(fontsize, zone_h_px / 1.2)
    fontsize = max(4, min(200, fontsize))
    fontsize_px = int(fontsize)

    print(f"Calculated font size: {fontsize_px}px")

    # Get actual text size with this font
    try:
        font_actual = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", fontsize_px)
    except:
        font_actual = ImageFont.load_default()

    try:
        tb = draw.textbbox((0, 0), text, font=font_actual)
        text_w = tb[2] - tb[0]
        text_h = tb[3] - tb[1]
    except:
        text_w = int(len(text) * fontsize_px * 0.6)
        text_h = fontsize_px

    print(f"Text size: {text_w} x {text_h} pixels")

    # Center position in zone
    text_x = px1 + max(0, (zone_w_px - text_w) / 2)
    text_y = py1 + max(0, (zone_h_px - text_h) / 2)

    print(f"Text top-left position in image: ({text_x:.1f}, {text_y:.1f})")

    # Baseline calculation
    try:
        ascent, descent = font_actual.getmetrics()
    except:
        ascent = int(fontsize_px * 0.8)
        descent = int(fontsize_px * 0.2)

    baseline_y_img = text_y + ascent
    print(f"Font metrics: ascent={ascent}, descent={descent}")
    print(f"Baseline Y in image coords: {baseline_y_img:.1f}")

    # Convert to PDF coordinates
    baseline_x_pdf = text_x * pdf_dims[0]["width"] / img.size[0]
    baseline_y_pdf = pdf_dims[0]["height"] - (
        baseline_y_img * pdf_dims[0]["height"] / img.size[1]
    )

    print(
        f"Baseline position in PDF coords: ({baseline_x_pdf:.1f}, {baseline_y_pdf:.1f})"
    )

    # Font size in points
    fontsize_pt = fontsize_px * 72 / 200
    print(f"Font size in points: {fontsize_pt:.1f}pt")

    # Test actual PDF insertion
    print("\n--- Testing PDF text insertion ---")
    doc_test = fitz.open()
    page_test = doc_test.new_page(
        width=pdf_dims[0]["width"], height=pdf_dims[0]["height"]
    )

    # Insert image background
    img_buf = io.BytesIO()
    img.save(img_buf, format="PNG")
    page_test.insert_image(page_test.rect, stream=img_buf.getvalue())

    # Insert our test text
    page_test.insert_text(
        (baseline_x_pdf, baseline_y_pdf),
        text,
        fontsize=fontsize_pt,
        fontfile="C:/Windows/Fonts/arial.ttf",
        fill_opacity=0,
        stroke_opacity=0,
    )

    # Save test PDF
    doc_test.save("coord_test.pdf")
    doc_test.close()
    print("Saved test PDF to coord_test.pdf")

    # Also create a visualization
    viz_img = img.copy()
    viz_draw = ImageDraw.Draw(viz_img)

    # Draw zone rectangle
    viz_draw.rectangle([px1, py1, px2, py2], outline="red", width=2)

    # Draw text baseline
    try:
        font_viz = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", fontsize_px)
    except:
        font_viz = ImageFont.load_default()

    viz_draw.text((text_x, text_y), text, fill="blue", font=font_viz)

    # Draw baseline marker
    baseline_x_img = text_x
    baseline_y_img_viz = text_y + ascent
    viz_draw.line(
        [
            (baseline_x_img, baseline_y_img_viz - 2),
            (baseline_x_img, baseline_y_img_viz + 2),
        ],
        fill="green",
        width=2,
    )
    viz_draw.line(
        [
            (baseline_x_img - 2, baseline_y_img_viz),
            (baseline_x_img + 2, baseline_y_img_viz),
        ],
        fill="green",
        width=2,
    )

    viz_img.save("coord_viz.png")
    print("Saved visualization to coord_viz.png")


if __name__ == "__main__":
    verify_zone_coordinates()
