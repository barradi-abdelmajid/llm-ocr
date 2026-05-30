"""Orchestrator: run Phase 1-4 sequentially with resume/skip support.

Usage:
  python run_all.py pdf_idk.pdf                      # run all 4 phases
  python run_all.py pdf_idk.pdf --start 3             # resume from phase 3
  python run_all.py pdf_idk.pdf --skip 3              # skip phase 3
  python run_all.py pdf_idk.pdf --start 2 --skip 3    # start at phase 2, skip 3
"""

import os, sys, argparse, subprocess, time, warnings
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"


def run_phase(script: str, args: list, phase_num: int, desc: str) -> bool:
    print(f"\n{'=' * 60}")
    print(f"  Phase {phase_num}: {desc}")
    print(f"{'=' * 60}")
    t0 = time.time()
    cmd = [sys.executable, script] + args
    result = subprocess.run(cmd)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(
            f"\n  Phase {phase_num} FAILED (exit code {result.returncode}, {elapsed:.1f}s)"
        )
        return False
    print(f"  Phase {phase_num} completed in {elapsed:.1f}s")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run all 4 phases of the OCR pipeline")
    parser.add_argument(
        "pdf", nargs="?", help="Path to PDF file (required for Phase 1)"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="./output",
        help="Final output directory (Phase 4)",
    )
    parser.add_argument(
        "--phases-dir", default="./phases", help="Intermediate phases directory"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help="Start from this phase (skip earlier ones)",
    )
    parser.add_argument(
        "--skip", type=int, nargs="*", default=[], help="Skip specific phase numbers"
    )
    parser.add_argument("--lm-studio", help="LM Studio URL (for Phase 3)")
    parser.add_argument(
        "--lm-studio-phase1",
        nargs="?",
        const="auto",
        default=None,
        metavar="URL",
        help="Use LM Studio for Phase 1 layout detection (auto-fallback when CUDA absent)",
    )
    parser.add_argument(
        "--lm-model",
        default="mistralai/ministral-3-3b",
        help="Model for LM Studio",
    )
    parser.add_argument(
        "--image-max-tokens",
        type=int,
        default=2048,
        help="Max tokens for image descriptions",
    )
    parser.add_argument(
        "--local-lm", help="Run Phase 3 locally via transformers (model ID)"
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Enable 4-bit quantization for Phase 3 (requires bitsandbytes + CUDA)",
    )
    parser.add_argument("--config", default="config.txt", help="Config file path")
    parser.add_argument("--dpi", type=int, default=200, help="PDF rendering DPI")
    parser.add_argument(
        "--invisible",
        action="store_true",
        help="Use invisible text overlay (render_mode=3)",
    )
    args = parser.parse_args()

    phases_dir = Path(args.phases_dir)
    phases_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    skip_set = set(args.skip)

    # Load config for shared parameters
    config = {}
    if os.path.isfile(args.config):
        for line in Path(args.config).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                config[k.strip()] = v.strip()

    pdf_path = args.pdf
    if not pdf_path and args.start == 1:
        print("Error: PDF path required when starting from Phase 1")
        sys.exit(1)

    lm_url = args.lm_studio or config.get("LM_STUDIO_HOST", "")
    lm_prompts = config.get("LM_PROMPTS_FILE", "lm_prompts.txt")
    lm_timeout = config.get("LM_STUDIO_TIMEOUT", "120")
    lm_max_tokens = config.get("LM_STUDIO_MAX_TOKENS", "2048")
    dpi = args.dpi

    phases = [
        (
            1,
            "Segmentation (PP-DocLayoutV3)",
            "phase1/segment.py",
            [pdf_path, "--output-dir", str(phases_dir), "--dpi", str(dpi)],
        ),
        (
            2,
            "OCR + Text Overlay (GLM-OCR)",
            "phase2/glm_overlay.py",
            [
                str(phases_dir / "phase1_zones.json"),
                "--output-dir",
                str(phases_dir),
                "--dpi",
                str(dpi),
            ],
        ),
        (
            3,
            "Image Analysis + Overlay (LM Studio)",
            "phase3/lm_overlay.py",
            [
                str(phases_dir / "phase2_zones.json"),
                "--output-dir",
                str(phases_dir),
                "--dpi",
                str(dpi),
                "--workers",
                "5",
                "--timeout",
                lm_timeout,
                "--max-tokens",
                lm_max_tokens,
            ],
        ),
        (
            4,
            "Build Final Markdown",
            "phase4/build_md.py",
            [
                str(phases_dir / "phase2_zones_v2.json"),
                "--output-dir", str(output_dir),
                "--docling", str(phases_dir / "phase2_ocrd_v2.pdf"),
            ],
        ),
    ]

    # Inject Phase 1 args (LM Studio layout)
    if args.lm_studio_phase1 is not None:
        phases[0][3].append(f"--lm-studio={args.lm_studio_phase1}")
        phases[0] = (
            1,
            "Segmentation (LM Studio)",
            phases[0][2],
            phases[0][3],
        )

    # Inject Phase 3 args
    if args.load_in_4bit:
        phases[2][3].append("--load-in-4bit")
    if args.local_lm:
        phases[2][3].extend(["--local-lm", args.local_lm])
    elif lm_url:
        phases[2][3].extend(["--lm-studio", lm_url])
        phases[2][3].extend(["--lm-model", args.lm_model])
        phases[2][3].extend(["--image-max-tokens", str(args.image_max_tokens)])
    if os.path.isfile(lm_prompts):
        phases[2][3].extend(["--prompts", lm_prompts])

    if args.invisible:
        phases[1][3].append("--invisible")  # Phase 2

    t_total = time.time()
    print(f"OCR Pipeline: 4 phases")
    print(f"  Start from: Phase {args.start}")
    print(f"  Skip: {list(skip_set) if skip_set else 'none'}")
    print(f"  Phases dir: {phases_dir}")
    print(f"  Output dir: {output_dir}")
    if args.lm_studio_phase1 is not None:
        print(f"  Phase 1 layout: LM Studio ({args.lm_studio_phase1})")
    if lm_url:
        print(f"  Phase 3 VLM: LM Studio ({lm_url})")
    print()

    for phase_num, desc, script, script_args in phases:
        if phase_num < args.start:
            print(f"  Phase {phase_num}: skipped (start={args.start})")
            continue
        if phase_num in skip_set:
            print(f"  Phase {phase_num}: skipped via --skip")
            continue

        script_path = Path(__file__).parent / script
        if not script_path.exists():
            print(f"Error: {script_path} not found")
            sys.exit(1)

        # Pre-phase GPU management
        gpu_phases = {2, 3}  # phases that load GPU models
        if phase_num in gpu_phases:
            try:
                import subprocess

                subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        "from gpu_utils import free_gpu_memory; free_gpu_memory(verbose=True)",
                    ],
                    timeout=30,
                )
            except Exception:
                pass

        # For Phase 1, need PDF; for later phases, check JSON exists
        if phase_num == 1:
            if not pdf_path or not os.path.isfile(pdf_path):
                print(f"Error: PDF not found: {pdf_path}")
                sys.exit(1)

        success = run_phase(str(script_path), script_args, phase_num, desc)
        if not success:
            print(f"\nPipeline ABORTED at Phase {phase_num}")
            sys.exit(1)

        # Post-phase GPU cleanup
        if phase_num in gpu_phases:
            try:
                import subprocess

                subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        "from gpu_utils import free_gpu_memory; free_gpu_memory(verbose=True)",
                    ],
                    timeout=30,
                )
            except Exception:
                pass

    elapsed = time.time() - t_total
    print(f"\n{'=' * 60}")
    msg = f"  Pipeline completed in {elapsed:.1f}s"
    if skip_set:
        msg += f" (skipped phase(s) {list(skip_set)})"
    print(msg)
    print(f"  Output: {output_dir / 'final.md'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
