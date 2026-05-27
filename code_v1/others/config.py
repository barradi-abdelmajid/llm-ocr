"""Configuration loader — reads config.txt and exports typed variables."""

import ast


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
PDF_FILENAME = _cfg.get("PDF_FILENAME", "pdf_idk.pdf")
OUTPUT_PATH = _cfg.get("OUTPUT_PATH", "output.md")
LLM_OUTPUT_PATH = _cfg.get("LLM_OUTPUT_PATH", "llm_output.md")

MIN_AREA = _cfg.get("MIN_AREA", 50)
INK_THRESHOLD = _cfg.get("INK_THRESHOLD", 220)
MIN_DRAWING_PIXELS = _cfg.get("MIN_DRAWING_PIXELS", 80)

Y_GAP = _cfg.get("Y_GAP", 18)
X_GAP = _cfg.get("X_GAP", 35)

BOX_MARGIN_PT = _cfg.get("BOX_MARGIN_PT", 4)

TEXT_DRAWING_OVERLAP = _cfg.get("TEXT_DRAWING_OVERLAP", 0.25)
TEXT_BOX_OVERLAP = _cfg.get("TEXT_BOX_OVERLAP", 0.30)

VBAND_Y_GAP = _cfg.get("VBAND_Y_GAP", 20)
VBAND_X_OVERLAP = _cfg.get("VBAND_X_OVERLAP", 0.08)

MAX_ZONE_HEIGHT_FRAC = _cfg.get("MAX_ZONE_HEIGHT_FRAC", 0.60)
GUARD_SPLIT_GAP = _cfg.get("GUARD_SPLIT_GAP", 12)

MERGE_Y_GAP = _cfg.get("MERGE_Y_GAP", 16)
MERGE_X_OVERLAP = _cfg.get("MERGE_X_OVERLAP", 0.03)
MERGE_X_GAP = _cfg.get("MERGE_X_GAP", 12)
MERGE_MAX_H = _cfg.get("MERGE_MAX_H", 500)

MERGE_H_GAP = _cfg.get("MERGE_H_GAP", 20)
MERGE_H_Y_OVERLAP = _cfg.get("MERGE_H_Y_OVERLAP", 0.15)

UNCOVERED_MIN_PIXELS = _cfg.get("UNCOVERED_MIN_PIXELS", 30)
UNCOVERED_MIN_AREA = _cfg.get("UNCOVERED_MIN_AREA", 50)

TINY_MIN_AREA = _cfg.get("TINY_MIN_AREA", 50)
TINY_MIN_W = _cfg.get("TINY_MIN_W", 10)
TINY_MIN_H = _cfg.get("TINY_MIN_H", 8)

DIVIDER_MAX_H = _cfg.get("DIVIDER_MAX_H", 8)
DIVIDER_MIN_W_FRAC = _cfg.get("DIVIDER_MIN_W_FRAC", 0.40)

COLOR_THRESHOLD = _cfg.get("COLOR_THRESHOLD", 40)
BLUE_BIAS = _cfg.get("BLUE_BIAS", 20)
COLORED_RATIO = _cfg.get("COLORED_RATIO", 0.25)

LM_STUDIO_HOST = _cfg.get("LM_STUDIO_HOST", "http://100.76.47.104:1234")
LM_VERIFY = _cfg.get("LM_VERIFY", True)
