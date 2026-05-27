# Circuit Theory PDF → Markdown Pipeline

A fully local French circuit-theory PDF-to-Markdown pipeline using PP-DocLayoutV3 for layout segmentation, GLM-OCR for text/formula/table recognition, and LM Studio (vision model) for image/chart analysis. Runs on a 6GB GPU machine.

## Pipeline Overview

```
pdf_idk.pdf
    │
    ▼
Phase 1 ── Layout Segmentation (PP-DocLayoutV3)
    │         Output: phase1_zones.json, phase1_viz.pdf (debug)
    ▼
Phase 2 ── OCR + Text Overlay (GLM-OCR)
    │         Output: phase2_zones.json, phase2_ocrd.pdf
    ▼
Phase 3 ── Image Analysis (LM Studio vision model)
    │         Output: phase3_zones.json, phase3_complete.pdf
    ▼
Phase 4 ── Markdown Builder
              Output: final.md
```

## Phase Details

### Phase 1: Layout Segmentation

Uses **PP-DocLayoutV3** from PaddlePaddle to detect layout regions (text, formulas, tables, images, charts, etc.) on each page.

- **Script**: `phase1/segment.py`
- **Input**: PDF file
- **Outputs**:
  - `phases/phase1_zones.json` — zone metadata with `bbox_norm` (0–1000 normalized coordinates), `label`, `task_type` (text / formula / table / skip)
  - `phases/phase1_viz.pdf` — visualization PDF with colored zone rectangles drawn on page images (green = text, blue = formula, red = table, yellow = skip)
- **Features**:
  - Maps detected labels to task types (e.g. `display_formula` → `formula`, `table` → `table`, `image` → `skip`)
  - Filters nested zones (smaller zones strictly contained in larger ones are removed)
  - Sorts zones in reading order (top-to-bottom, left-to-right)
  - Abandons irrelevant regions (headers, footers, page numbers, etc.)

### Phase 2: OCR + PDF Text Overlay

Uses **GLM-OCR** (zai-org/GLM-OCR) to recognize text, formulas, and tables from each cropped zone. Overlays selectable (invisible) text into the PDF at the correct zone positions.

- **Script**: `phase2/glm_overlay.py`
- **Input**: `phases/phase1_zones.json`
- **Outputs**:
  - `phases/phase2_ocrd.pdf` — PDF with:
    - Background page image (200 DPI render of original PDF)
    - Visible text drawn on the image at zone positions (PIL, Arial font)
    - Invisible selectable PDF text (`insert_textbox`, `render_mode=3`) at matching coordinates
  - `phases/phase2_zones.json` — zones with OCR `content` field added
- **Key Details**:
  - Uses task-specific prompts: `"Text Recognition:"`, `"Formula Recognition:"`, `"Table Recognition:"`
  - Newlines stripped from OCR output (`\\n` → space) so formulas like `$$ ... $$` appear on a single line
  - Font size calculated via PIL `textlength` for accuracy; converted from pixel space to PDF points (`fontsize_pt = fontsize_px × 72 / dpi`)
  - `insert_textbox` return value checked (PyMuPDF 1.27+ returns float; negative = overflow); automatic retry with 0.65× font reduction on overflow
  - Coordinate system: pixel coords from `int(bbox_norm * img_dim / 1000)`, Y-flipped for PDF (PDF origin is bottom-left)
  - Supports `--invisible` flag to skip the visible PIL text overlay
- **Dependencies**: transformers, torch, Pillow, PyMuPDF, ~6GB VRAM for GLM-OCR

### Phase 3: Image/Chart Analysis

Sends image and chart zones to a **vision-capable LM** (LM Studio or local transformers) for description. Processes only zones with `task_type: "skip"` that have labels `"image"` or `"chart"`.

- **Script**: `phase3/lm_overlay.py`
- **Input**: `phases/phase2_zones.json`
- **Outputs**:
  - `phases/phase3_complete.pdf` — PDF with image descriptions overlaid
  - `phases/phase3_zones.json` — zones with LM-generated content
- **Modes**:
  - **LM Studio** (default): Sends base64-encoded JPEG crops to LM Studio's `/chat/completions` API. Requires a vision model loaded (e.g. LLaVA, Qwen-VL — **not** text-only models like Mistral)
  - **Local** (`--local-lm`): Loads a HuggingFace vision-language model via `transformers` (same processor pattern as GLM-OCR). Specify model ID e.g. `llava-hf/llava-1.5-7b-hf`
- **Features**:
  - Circuit region grouping: nearby image/chart zones on the same page are grouped for context-aware prompting
  - First zone in group gets `"image"` prompt, subsequent zones get `"image_modification"` prompt
  - Custom prompts via `lm_prompts.txt` file (formatted with `# prompt_name` headers)
  - Parallel workers for LM Studio, sequential for local mode
  - Fetch prompts path from `config.txt` (`LM_PROMPTS_FILE`)

### Phase 4: Markdown Builder

Assembles recognized content into a clean Markdown file, sorted by page and reading order.

- **Script**: `phase4/build_md.py`
- **Input**: `phases/phase3_zones.json` (or `phase2_zones.json` if Phase 3 skipped)
- **Output**: `output/final.md`
- **Features**:
  - Zone type ordering: circuits first, then formulas, text, images, charts, headings, tables, references, etc.
  - Content formatted with appropriate Markdown syntax (code blocks for formulas, etc.)
  - Optional Docling integration (`--docling` flag) for alternative OCR via Docling

## Orchestrator

The `run_all.py` script runs all 4 phases sequentially with resume/skip support.

```powershell
python run_all.py pdf_idk.pdf                    # run all phases
python run_all.py pdf_idk.pdf --start 3           # resume from phase 3
python run_all.py pdf_idk.pdf --skip 3            # skip phase 3
python run_all.py pdf_idk.pdf --start 2 --skip 3  # start at phase 2, skip 3
python run_all.py pdf_idk.pdf --invisible         # invisible text overlay only
```

Key flags:
- `--start N` — begin from phase N (skips earlier phases)
- `--skip N [M ...]` — skip specific phase(s)
- `--invisible` — use invisible text overlay (`render_mode=3`) in phases 2 and 3
- `--local-lm MODEL_ID` — run Phase 3 locally via transformers
- `--lm-studio URL` — LM Studio API URL (overrides config.txt)
- `--dpi N` — PDF rendering DPI (default 200)

## Configuration

`config.txt` supports:
```
LM_STUDIO_HOST=http://localhost:1234
LM_PROMPTS_FILE=lm_prompts.txt
LM_STUDIO_TIMEOUT=120
LM_STUDIO_MAX_TOKENS=512
```

## Output Structure

```
phases/
├── phase1_zones.json      Zone metadata (bounding boxes, labels)
├── phase1_viz.pdf         Visualization with colored zone rects
├── phase2_zones.json      Zones + OCR content
├── phase2_ocrd.pdf        OCR'd PDF with selectable text
├── phase3_zones.json      Zones + LM descriptions
├── phase3_complete.pdf    Final PDF with image descriptions
├── diagnose_invisible.pdf Coordinate debug output
├── diag_final.pdf         Clean coordinate verification
└── coord_test.pdf         Coordinate conversion test

output/
└── final.md               Final Markdown output
```

## Dependencies

- Python 3.10+
- PyTorch (CUDA)
- transformers + accelerate
- Pillow, PyMuPDF (fitz)
- PaddlePaddle (via docling for PP-DocLayoutV3)
- GLM-OCR (`zai-org/GLM-OCR`)
- pynvml (GPU monitoring)

## Coordinate System

All zone positions use `bbox_norm` — normalized coordinates in the range [0, 1000] representing percentage positions on the page. Conversion to pixel coordinates (for PIL rendering) and PDF points (for `insert_textbox`) follows:

```python
# Pixel coords on 200 DPI image
px = int(bbox_norm[0] * img_width / 1000)
py = int(bbox_norm[1] * img_height / 1000)

# PDF coords (Y-flip: PDF origin is bottom-left)
pdf_x = px * pdf_width / img_width
pdf_y = (img_height - py) * pdf_height / img_height  # Y-flip
```

## Notes

- Phase 2 re-runs GLM-OCR on each invocation (no cached OCR mode yet)
- Phase 3 requires a vision-capable model in LM Studio; Mistral 3B (text-only) will produce `"ERROR: Cannot read 'image.png'"`
- The 6GB GPU limit constrains model choice; GLM-OCR fits comfortably
- All intermediate files use the same PDF path reference; the pipeline can be resumed after partial completion
