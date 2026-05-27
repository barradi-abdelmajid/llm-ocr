"""Output writers — Obsidian markdown and structured LLM report."""

from config import PDF_FILENAME, OUTPUT_PATH, LLM_OUTPUT_PATH, MIN_DRAWING_PIXELS


def write_obsidian_output(pages, lines):
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_llm_report(pages, llm_zones):
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
        f"**Page size:** {pw} x {ph} pt (Y=0 at bottom)"
    )
    lines.append("")

    for page in pages:
        pn = page["num"]
        pz = sorted(
            [z for z in llm_zones if z["page"] == pn],
            key=lambda z: -z["y_top"],
        )
        ink = page.get("ink_components", [])
        lines.append(f"---")
        lines.append(f"## Page {pn}")
        lines.append(
            f"Zones: {len(pz)} | Ink components: {len([c for c in ink if c['pixels'] >= MIN_DRAWING_PIXELS])}"
        )
        lines.append("")

        lines.append("| # | Label | rect (L,B,R,T) | W | H |")
        lines.append("|---|-------|----------------|---|---|")
        for i, z in enumerate(pz, 1):
            lines.append(
                f"| {i} | {z['label']} | {fmt_rect(z)} | {z['w']:.0f} | {z['h']:.0f} |"
            )
        lines.append("")

        sorted_bot = sorted(pz, key=lambda z: z["y_bot"])
        gaps = []
        for i in range(len(sorted_bot) - 1):
            gap = gap_size(sorted_bot[i + 1], sorted_bot[i])
            if gap > 5:
                gaps.append(
                    f"gap {gap:.0f}pt between zone at Y={sorted_bot[i]['y_top']:.0f}--{sorted_bot[i + 1]['y_bot']:.0f}"
                )
        if gaps:
            lines.append("### Vertical gaps between zones")
            for g in gaps:
                lines.append(f"- {g}")
            lines.append("")

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
                                f"horizontal gap {hgap:.0f}pt at Y={gy_bot:.0f}--{gy_top:.0f} (has ink)"
                            )
                            break

        if edges:
            lines.append("### Uncovered page margins")
            for e in edges:
                lines.append(f"- {e}")
            lines.append("")

        if gaps or h_gaps:
            lines.append("### Potential issues")
            for g in gaps:
                lines.append(f"- {g}")
            for h in h_gaps:
                lines.append(f"- {h}")
            lines.append("")

    lines.append("---")
    lines.append("## Review instructions")
    lines.append(
        "Check each page for: 1) zones that are too large, "
        "2) zones that are too small, "
        "3) vertical gaps with ink not captured, "
        "4) horizontal gaps between zones that may contain missed strokes."
    )
    lines.append("")

    with open(LLM_OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    issue_count = len(gaps) + len(h_gaps) + len(edges)
    print(
        f"Wrote {LLM_OUTPUT_PATH} ({len(llm_zones)} zones, {issue_count} flagged gaps/margins)"
    )
