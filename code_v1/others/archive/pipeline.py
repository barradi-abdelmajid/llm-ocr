"""Main pipeline — orchestrates extraction, zoning, and output."""

import sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from config import (
    PDF_PATH,
    PDF_FILENAME,
    OUTPUT_PATH,
    LLM_OUTPUT_PATH,
    LM_VERIFY,
    VBAND_Y_GAP,
    VBAND_X_OVERLAP,
    MERGE_Y_GAP,
    MERGE_X_OVERLAP,
    MERGE_X_GAP,
    MERGE_MAX_H,
    MERGE_H_GAP,
    MERGE_H_Y_OVERLAP,
    TEXT_DRAWING_OVERLAP,
    TEXT_BOX_OVERLAP,
    UNCOVERED_MIN_PIXELS,
    UNCOVERED_MIN_AREA,
    MIN_DRAWING_PIXELS,
    GUARD_SPLIT_GAP,
    MAX_ZONE_HEIGHT_FRAC,
)

from extract import (
    extract_notability_objects,
    extract_embedded_images,
    find_ink_components,
    find_drawing_regions,
    find_box_zones,
    overlap_ratio as extract_overlap_ratio,
)

from zoning import (
    y_cluster_pdf,
    x_split_cluster,
    get_docling_layout,
    classify,
    guess_drawing_label,
    merge_overlapping,
    remove_contained,
    merge_vertical_bands,
    merge_neighbors,
    merge_drawing_text,
    filter_divider_lines,
    filter_tiny_zones,
    clamp_to_page,
    find_uncovered_ink,
    relabel_by_ink_color,
    guard_split_oversized_zones,
)

from llm_client import check_lm_studio, llm_analyze_report
from output import write_obsidian_output, write_llm_report


def main():
    pages = extract_notability_objects(PDF_PATH)
    print(f"Extracted {sum(len(p['objects']) for p in pages)} text objects")

    extract_embedded_images(PDF_PATH, pages)
    find_ink_components(PDF_PATH, pages)
    find_drawing_regions(pages)
    find_box_zones(PDF_PATH, pages)

    print("Running Docling layout analysis...")
    docling_items = get_docling_layout(PDF_PATH)

    lm_ok = False
    if LM_VERIFY:
        print("Checking LM Studio...")
        if check_lm_studio():
            lm_ok = True
        else:
            print("  WARNING: LM Studio not running. Skipping LLM gap analysis.")

    lines = []
    llm_zones = []

    for page in pages:
        page_items = [i for i in docling_items if i["page"] == page["num"]]
        pw = page["width"]

        # --- build text zones with horizontal splitting ---
        text_zones = []
        for c in y_cluster_pdf(page["objects"]):
            for sub_c in x_split_cluster(c):
                text_zones.append(
                    {
                        "x0": min(o["x0"] for o in sub_c),
                        "x1": max(o["x1"] for o in sub_c),
                        "y_bot": min(o["y_bot"] for o in sub_c),
                        "y_top": max(o["y_top"] for o in sub_c),
                        "label": classify(sub_c, page_items),
                    }
                )

        # --- merge vertical bands (group lines in same column) ---
        text_zones = merge_vertical_bands(
            text_zones, y_gap=VBAND_Y_GAP, x_overlap_ratio=VBAND_X_OVERLAP
        )

        text_zones = guard_split_oversized_zones(
            text_zones, page["objects"], page_items, page["height"]
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
                if (
                    extract_overlap_ratio(tz, dz) > TEXT_DRAWING_OVERLAP
                    and tz["label"] != "formula"
                ):
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
            filtered_drawings = []
            for dz in drawing_zones:
                suppressed = False
                for bz in box_zones:
                    if extract_overlap_ratio(dz, bz) > 0.4:
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
                    if extract_overlap_ratio(tz, bz) > TEXT_BOX_OVERLAP:
                        bz["x0"] = min(bz["x0"], tz["x0"])
                        bz["x1"] = max(bz["x1"], tz["x1"])
                        bz["y_bot"] = min(bz["y_bot"], tz["y_bot"])
                        bz["y_top"] = max(bz["y_top"], tz["y_top"])
                        swallowed = True
                        break
                if not swallowed:
                    filtered.append(tz)
            keep_text = filtered + box_zones

        # --- merge & dedup zones ---
        all_zones = keep_text + image_zones + drawing_zones
        all_zones = merge_overlapping(all_zones)
        all_zones = remove_contained(all_zones)

        # --- filter divider lines ---
        all_zones = filter_divider_lines(all_zones, pw)

        # --- merge neighbors (vertical, height-guarded) ---
        all_zones = merge_neighbors(
            all_zones,
            y_gap=MERGE_Y_GAP,
            x_overlap_ratio=MERGE_X_OVERLAP,
            max_h=MERGE_MAX_H,
            x_gap=MERGE_X_GAP,
        )

        # --- merge drawing with adjacent text (side-by-side) ---
        all_zones = merge_drawing_text(
            all_zones,
            x_gap=MERGE_H_GAP,
            y_overlap_ratio=MERGE_H_Y_OVERLAP,
        )

        # --- find & fill uncovered ink (pixel-level) ---
        uncovered = find_uncovered_ink(
            PDF_PATH,
            page["num"],
            all_zones,
            page["height"],
            min_pixels=UNCOVERED_MIN_PIXELS,
        )
        missed_zones = []
        for blob in uncovered:
            if (blob["x1"] - blob["x0"]) * (
                blob["y_top"] - blob["y_bot"]
            ) < UNCOVERED_MIN_AREA:
                continue
            missed_zones.append({**blob, "label": "notability-missed"})
        all_zones += missed_zones
        all_zones = remove_contained(all_zones)

        # --- clamp to page bounds ---
        all_zones = clamp_to_page(all_zones, pw, page["height"])

        # --- filter tiny zones ---
        all_zones = filter_tiny_zones(all_zones)

        page["zones"] = all_zones

    # --- color-based relabeling ---
    print("Checking ink color for label refinement...")
    relabel_by_ink_color(PDF_PATH, pages)

    # --- output zones ---
    for page in pages:
        for z in page["zones"]:
            rect = f"{z['x0']:.0f},{z['y_bot']:.0f},{z['x1']:.0f},{z['y_top']:.0f}"
            lines.append(
                f"![[{PDF_FILENAME}#page={page['num']}&rect={rect}|{PDF_FILENAME}, p.{page['num']}]]  <!-- {z['label']} -->"
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

    write_obsidian_output(pages, lines)

    print(f"\nWrote {len(llm_zones)} zones to {OUTPUT_PATH}")
    write_llm_report(pages, llm_zones)

    # Per-page zone counts
    for page in pages:
        pz = [z for z in llm_zones if z["page"] == page["num"]]
        print(f"  Page {page['num']}: {len(pz)} zones")

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
