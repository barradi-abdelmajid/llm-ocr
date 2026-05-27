"""Debug: draw zone boundaries (blue) + extracted text positions (red)
with Y-coordinate annotations to diagnose vertical misalignment."""

import json
import fitz

with open("phases/phase2_zones.json", encoding="utf-8") as f:
    data = json.load(f)

zones = data["zones"]
pdf_dims = data["pdf_dims"]

doc = fitz.open("phases/phase2_ocrd.pdf")

for page_idx in range(min(1, len(doc))):
    page = doc[page_idx]
    pw = pdf_dims[page_idx]["width"]
    ph = pdf_dims[page_idx]["height"]

    # Get text blocks with positions
    text_blocks = page.get_text("blocks")

    # Collect zone rects for this page
    page_zones = [z for z in zones if z["page"] == page_idx]

    print(
        f"=== Page {page_idx + 1}: {len(page_zones)} zones, {len(text_blocks)} text blocks ==="
    )
    print()

    # Draw all zones
    for z in page_zones:
        b = z["bbox_norm"]
        x1 = b[0] * pw / 1000
        x2 = b[2] * pw / 1000
        y1 = (1000 - b[3]) * ph / 1000
        y2 = (1000 - b[1]) * ph / 1000

        has_content = bool(z.get("content", "").strip())
        color = (0, 1, 0) if has_content else (0, 0, 1)  # green=content, blue=empty
        rect = fitz.Rect(x1, y1, x2, y2)
        page.draw_rect(rect, color=color, width=0.5)

        # Label zone with its Y range
        label = f"Z y={y1:.0f}-{y2:.0f}"
        page.insert_text(fitz.Point(x1, y2 + 10), label, fontsize=6, color=color)

    # Draw text block boundaries in red
    for tb in text_blocks:
        x0, y0, x1, y1, text, *_ = tb
        if not text.strip():
            continue
        rect = fitz.Rect(x0, y0, x1, y1)
        page.draw_rect(rect, color=(1, 0, 0), width=0.5)
        txt = text.strip()[:30].replace("\n", " ")
        label = f"T y={y0:.0f}-{y1:.0f}"
        page.insert_text(fitz.Point(x0, y0 - 3), label, fontsize=5, color=(1, 0, 0))

    # Print comparison table
    print(
        f"{'Text pos (Y range)':<25} {'Zone pos (Y range)':<25} {'Delta':<10} {'Label':<20} {'Text':<40}"
    )
    print("-" * 120)

    # For each text block, find closest zone
    for tb in text_blocks:
        x0, y0, x1, y1, text, *_ = tb
        text_content = text.strip()[:40].replace("\n", " ")
        if not text_content:
            continue

        # Find zone that contains or is closest to this text block
        best_zone = None
        best_overlap = 0
        for z in page_zones:
            b = z["bbox_norm"]
            zx1 = b[0] * pw / 1000
            zx2 = b[2] * pw / 1000
            zy1 = (1000 - b[3]) * ph / 1000
            zy2 = (1000 - b[1]) * ph / 1000

            # Check Y overlap
            overlap = max(0, min(y1, zy2) - max(y0, zy1))
            if overlap > best_overlap:
                best_overlap = overlap
                best_zone = (
                    zx1,
                    zy1,
                    zx2,
                    zy2,
                    z.get("label", "?"),
                    z.get("content", "")[:30],
                )

        if best_zone:
            zx1, zy1, zx2, zy2, zlabel, zcontent = best_zone
            dy_top = y1 - zy2  # text top - zone top (positive = text above zone)
            dy_bot = (
                y0 - zy1
            )  # text bottom - zone bottom (positive = text above zone bottom)
            print(
                f"T:({y0:.0f},{y1:.0f}){'':10s} Z:({zy1:.0f},{zy2:.0f}){'':10s} d_top={dy_top:+.0f} d_bot={dy_bot:+.0f}  {zlabel:<20s} {text_content:<40s}"
            )

doc.save("debug_alignment.pdf", deflate=True)
doc.close()
print("\nSaved debug_alignment.pdf")
