"""Phase 4: Build final Markdown from zones JSON.
Builds clean markdown from recognized zone content.
Optionally tries Docling on the OCR'd PDF (see --docling flag).

Usage:
  python phase4/build_md.py phases/phase3_zones.json -o ./output
  python phase4/build_md.py phases/phase3_zones.json -o ./output --docling  # try Docling too
"""

import os, json, time, sys, argparse, re
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


TASK_ORDER = {
    "circuit": 0,
    "formula": 1,
    "text": 2,
    "image": 3,
    "chart": 4,
    "heading": 5,
    "table": 6,
    "reference": 7,
    "abstract": 8,
    "title": 9,
    "caption": 10,
    "seal": 11,
    "formula_num": 12,
}


def sort_zones_on_page(zones):
    return sorted(
        zones,
        key=lambda z: (
            TASK_ORDER.get(z.get("label", "text"), 99),
            z["bbox_norm"][1],
            z["bbox_norm"][0],
        ),
    )


def format_final_markdown(zones, pdf_dims):
    lines = []
    for page_idx in range(len(pdf_dims)):
        page_zones = [z for z in zones if z["page"] == page_idx]
        if not page_zones:
            continue
        page_zones = sort_zones_on_page(page_zones)
        lines.append(f"## Page {page_idx + 1}")
        lines.append("")
        for z in page_zones:
            content = z.get("content", "").strip()
            if content:
                content_lines = content.splitlines()
                content_lines = [
                    cl for cl in content_lines if not cl.strip().startswith("```")
                ]
                content = "\n".join(content_lines).strip()
                if content:
                    lines.append(content)
                    lines.append("")
    return "\n".join(lines)


def fix_markdown_strict(text):
    lines = text.splitlines()
    result = []
    in_code_block = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
        if "$$" not in line:
            s_count = line.count("$")
            if s_count % 2 != 0 and s_count > 0:
                line = line + " $"
        line = re.sub(r"\\frac\s+\{", r"\\frac{", line)
        line = re.sub(r"\}\s*\{", r"}{", line)
        line = re.sub(r"\\([a-zA-Z]+)\s+\{", r"\\\1{", line)
        result.append(line)
    full = "\n".join(result)
    dd_count = full.count("$$")
    if dd_count % 2 != 0:
        full += "\n$$"
    return full


def check_markdown_syntax(text):
    warnings_list = []
    dollar_count = text.count("$$")
    if dollar_count % 2 != 0:
        warnings_list.append(f"Unbalanced $$: {dollar_count} delimiters")
    for i, line in enumerate(text.splitlines(), 1):
        s_count = line.count("$")
        if s_count % 2 != 0 and "$$" not in line:
            warnings_list.append(f"Line {i}: unbalanced $ (odd count)")
    return warnings_list


def try_docling(pdf_path):
    try:
        from docling.document_converter import DocumentConverter

        conv = DocumentConverter()
        result = conv.convert(pdf_path)
        doc = result.document
        return doc.export_to_markdown()
    except Exception as e:
        print(f"  Docling: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Phase 4: Build Final Markdown")
    parser.add_argument("zones_json", help="Phase 3 zones JSON file")
    parser.add_argument(
        "--output-dir", "-o", default="./output", help="Output directory"
    )
    parser.add_argument(
        "--docling", help="Optional: try Docling on this PDF for comparison"
    )
    args = parser.parse_args()

    if not os.path.isfile(args.zones_json):
        print(f"Error: zones JSON not found: {args.zones_json}")
        sys.exit(1)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(args.zones_json, encoding="utf-8") as f:
        data = json.load(f)

    zones = data["zones"]
    pdf_dims = data.get(
        "pdf_dims", [{"width": 612, "height": 792}] * data.get("num_pages", 1)
    )

    print(f"[Phase 4] Building final markdown from zones JSON")
    print(f"  Zones: {len(zones)}, Pages: {len(pdf_dims)}")
    t0 = time.time()

    md = format_final_markdown(zones, pdf_dims)
    md = fix_markdown_strict(md)

    if args.docling and os.path.isfile(args.docling):
        print(f"  Trying Docling on {args.docling}...")
        docling_md = try_docling(args.docling)
        if docling_md:
            if len(docling_md) > len(md) * 0.5:
                print(f"  Docling output available ({len(docling_md)} bytes)")
                (out / "final_docling_raw.md").write_text(docling_md, encoding="utf-8")
                print(f"  Saved Docling raw output for reference")

    final_path = out / "final.md"
    final_path.write_text(md, encoding="utf-8")

    md_warnings = check_markdown_syntax(md)
    if md_warnings:
        print(f"  Markdown warnings ({len(md_warnings)}):")
        for w in md_warnings:
            print(f"    - {w}")
    else:
        print(f"  Markdown syntax: OK")

    print(f"  Saved {final_path} ({len(md)} bytes)")
    print(f"[Phase 4] Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
