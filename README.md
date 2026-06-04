# still in développement. nearly ready

# Circuit Theory PDF → Markdown Pipeline

Local French circuit-theory PDF-to-Markdown pipeline. Runs on an RTX 4050 6GB laptop (iGPU for display, dGPU for compute).

## Stack

| Component | Role |
|-----------|------|
| **PP-DocLayoutV3** (PaddlePaddle) | Phase 1 — Layout segmentation: detects text, formulas, tables, images, charts per page |
| **GLM-OCR** (`zai-org/GLM-OCR` via `transformers`) | Phase 2 — OCR: recognizes text, formulas (`$$...$$`), and tables from cropped zones |
| **LM Studio** (local inference server) | Phase 3 — Vision-language analysis: describes circuit diagrams, charts, images via OpenAI-compatible API |
| **Mistral 3B** (LM Studio, default) | Phase 3 vision model for circuit/chart descriptions |
| **Docling** | Phase 4 — Converts overlaid PDF to Markdown with `docling` |
| **PyTorch 2.11+cu128** | GPU backend for GLM-OCR |
| **bitsandbytes** | 4-bit quantization (optional, for local VLM) |
| **PyMuPDF (fitz)** | PDF manipulation, text overlay, page rendering |
| **Python 3.13** | Runtime |

## Pipeline

```
pdf → Phase 1 (PP-DocLayoutV3) → Phase 2 (GLM-OCR) → Phase 3 (LM Studio) → Phase 4 (Docling) → final.md
```

Phases 1–2 run on GPU (CUDA). Phase 3 calls LM Studio API (local or remote). Phase 4 uses Docling.

## Usage

```powershell
python run_all.py document.pdf                    # full pipeline
python run_all.py document.pdf --skip 3            # skip VLM analysis
python run_all.py document.pdf --start 2           # resume from phase 2
python run_all.py document.pdf --local-lm Qwen/Qwen3-VL-2B-Instruct --load-in-4bit
```

Configure LM Studio in `config.txt`:
```
LM_STUDIO_HOST = http://100.76.47.104:1234/v1
```

Customize VLM prompts in `lm_prompts.txt`.

## Phase Details

### Phase 1 — Layout Segmentation (`segment.py`)
- Detects 20+ region types (text, display_formula, table, figure, etc.)
- Maps labels → task types (`text`, `formula`, `table`, `skip`)
- Filters nested zones, sorts in reading order
- Output: `phase1_zones.json` (bounding boxes + labels), `phase1_viz.pdf` (debug overlay)

### Phase 2 — OCR + Text Overlay (`glm_overlay.py`)
- Loads GLM-OCR (`zai-org/GLM-OCR`) via `GlmOcrForConditionalGeneration` (bfloat16, ~3GB VRAM)
- Recognizes each non-skip zone (task-specific prompts)
- Multi-line zones split into sub-zones (except formulas → newlines stripped)
- Overlays invisible selectable text (`render_mode=3`) into PDF
- Output: `phase2_ocrd_v2.pdf`, `phase2_zones_v2.json`

### Phase 3 — Image Analysis (`lm_overlay.py`)
- Sends image/chart zones to LM Studio (Mistral 3B) for description
- Groups nearby zones on same page: first gets "image" prompt, rest get "image_modification"
- Parallel workers (default 5) for LM Studio
- Overlays descriptions into PDF at zone positions
- Falls back to original PDF if Phase 2 PDF not found
- Output: `phase3_complete.pdf`, `phase3_zones.json`

### Phase 4 — Markdown Export (`build_md.py`)
- Runs Docling on `phase2_ocrd_v2.pdf`
- Inserts `<!-- image -->` placeholders for zones not annotated by Phase 3
- Output: `output/final.md`

## Output

```
phases/
├── phase1_zones.json / phase1_viz.pdf
├── phase2_zones_v2.json / phase2_ocrd_v2.pdf
├── phase3_zones.json / phase3_complete.pdf
output/
└── final.md
```

## Key Design Decisions

- **Y-coordinate in PDF**: `insert_text()` treats Y as from top of page (screen coords), not bottom (PDF coords). `raw_Y = page_height - passed_Y`.
- **Font size**: Width-based (zone_w / text_length) for natural horizontal fit, centered vertically in zone.
- **Multi-line text**: Zones with >1 content line are split evenly into sub-zones (except formulas).
- **Formulas**: Newlines stripped (`\n` → space) for single-line `$$...$$` display.
- **HF Hub**: Offline mode + warnings suppressed via env vars in `load_glm_ocr()`.
