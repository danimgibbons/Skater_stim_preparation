"""Command-line interface: options, validation, and stage dispatch."""

import argparse
import re
from pathlib import Path

from .background import clean_background
from .camera_cut import create_camera_switch_video
from .config import DEFAULT_STAGES, STAGES
from .export_masks import export_frame_masks
from .mask_overlay import create_mask_overlay
from .preprocess import cut_big_chunks, cut_small_chunks
from .static_overlays import render_static_overlays
from .video import configure_data_paths, show_configured_paths


STAGE_RUNNERS = {
    "cut_big": cut_big_chunks,
    "cut_small": cut_small_chunks,
    "clean_bg": clean_background,
    "export_masks": export_frame_masks,
    "mask_overlay": create_mask_overlay,
    "static_overlays": render_static_overlays,
    "camera_cut": create_camera_switch_video,
}


def run_stage(stage: str, args: argparse.Namespace) -> None:
    """Dispatch one named pipeline stage."""
    try:
        runner = STAGE_RUNNERS[stage]
    except KeyError as exc:
        raise ValueError(f"Unknown stage: {stage}") from exc
    runner(args)


def parse_kernel(value: str) -> list[int]:
    """Parse kernel sizes written as WIDTH,HEIGHT."""
    parts = [int(part) for part in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Kernel values must be WIDTH,HEIGHT")
    return parts


def parse_static_overlay_set_name(value: str) -> str:
    """Parse a safe static overlay set folder name."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise argparse.ArgumentTypeError(
            "Static overlay set names may only contain letters, numbers, dots, "
            "underscores, and hyphens"
        )
    return value


def parse_output_label(value: str) -> str:
    """Parse a safe final output label."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise argparse.ArgumentTypeError(
            "Output labels may only contain letters, numbers, dots, "
            "underscores, and hyphens"
        )
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare synchronized multi-camera videos for MEG stimuli.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Optional JSON config with data_root, input_dir, and output_dir. "
            "Defaults to SKATER_PIPELINE_CONFIG or pipeline_config.json when present."
        ),
    )
    parser.add_argument(
        "--write-config",
        type=Path,
        default=None,
        help="Write a starter path config JSON file and exit.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Base folder for relative input/output paths, such as a mounted server folder.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Input folder. Relative paths are resolved under --data-root when set.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output folder. Relative paths are resolved under --data-root when set.",
    )
    parser.add_argument(
        "--show-paths",
        action="store_true",
        help="Print the effective config/input/output paths and exit.",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=STAGES + ["all", "preprocess"],
        default=None,
        help="Stages to run. Use preprocess for cut_big followed by cut_small.",
    )
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--lead-in-seconds", type=float, default=4)
    parser.add_argument("--small-duration", type=float, default=None)
    parser.add_argument("--small-duration-min", type=float, default=120)
    parser.add_argument("--small-duration-max", type=float, default=180)
    parser.add_argument("--chunk-pattern", default="big*_small*.mp4")
    parser.add_argument("--clean-crf", type=int, default=0)
    parser.add_argument("--clean-preset", default="medium")
    parser.add_argument("--mask-threshold", type=int, default=50)
    parser.add_argument("--blur-kernel", type=parse_kernel, default=[21, 21])
    parser.add_argument("--close-kernel", type=parse_kernel, default=[3, 3])
    parser.add_argument("--color-tolerance", type=int, default=10)
    parser.add_argument(
        "--static-overlay-set",
        type=parse_static_overlay_set_name,
        default=None,
        help=(
            "Named folder under input/static_overlays to use for static overlays, "
            "such as conditionA. If omitted, input/static_overlays.json is used."
        ),
    )
    parser.add_argument(
        "--no-static-overlays",
        action="store_true",
        help=(
            "During camera_cut, ignore default static-overlaid clips and use "
            "overlaid/cleaned/small sources instead. Cannot be combined with "
            "--static-overlay-set."
        ),
    )
    parser.add_argument(
        "--final-label",
        type=parse_output_label,
        default=None,
        help="Append a safe label to camera_cut output filenames.",
    )
    parser.add_argument(
        "--camera-cut-mode",
        choices=("random", "cycle"),
        default="random",
        help=(
            "Camera selection strategy for camera_cut. random keeps the existing "
            "seeded random switching; cycle uses camera folders in sorted order."
        ),
    )
    parser.add_argument(
        "--camera-cut-chunks",
        nargs="+",
        default=None,
        help=(
            "Exact small chunks for camera_cut, such as big1_small2 big1_small3. "
            "Defaults to all chunks matching --chunk-pattern."
        ),
    )
    parser.add_argument(
        "--big-index",
        type=int,
        default=None,
        help="Run camera_cut for one big chunk when used with --small-index.",
    )
    parser.add_argument(
        "--small-index",
        type=int,
        default=None,
        help="Run camera_cut for one small chunk when used with --big-index.",
    )
    parser.add_argument("--cut-frames", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    configure_data_paths(args, parser)
    if args.show_paths:
        show_configured_paths(args)
        raise SystemExit(0)

    steps = args.steps or DEFAULT_STAGES
    if "all" in steps:
        steps = DEFAULT_STAGES
    if "preprocess" in steps:
        steps = [step for step in steps if step != "preprocess"]
        steps = ["cut_big", "cut_small", *steps]

    if args.small_index is not None and args.big_index is None:
        parser.error("--small-index requires --big-index")

    if "camera_cut" in steps:
        if args.static_overlay_set and args.no_static_overlays:
            parser.error("--static-overlay-set cannot be combined with --no-static-overlays")
        if args.camera_cut_chunks and (
            args.big_index is not None or args.small_index is not None
        ):
            parser.error("--camera-cut-chunks cannot be combined with --big-index/--small-index")
        if (args.big_index is None) != (args.small_index is None):
            parser.error("--big-index and --small-index must be used together")

    for index, step in enumerate(steps, start=1):
        print(f"\n=== Stage {index}/{len(steps)}: {step} ===")
        run_stage(step, args)
