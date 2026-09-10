"""Shared file-discovery and FFmpeg helpers used by pipeline stages."""

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
from pathlib import Path

from .config import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_INPUT_DIR,
    DEFAULT_OUTPUT_DIR,
    STATIC_OVERLAY_SUFFIX,
    VIDEO_EXTENSIONS,
    CropBox,
    Normalization,
    StaticOverlay,
    load_path_config,
    resolve_data_path,
    write_config_template,
)

def crop_filter(crop: CropBox) -> str:
    """Return an FFmpeg crop filter that preserves odd crop origins exactly."""
    return (
        f"crop=w={crop.width}:h={crop.height}:"
        f"x={crop.x}:y={crop.y}:exact=1"
    )


def normalization_filter(normalization: Normalization) -> str:
    """Return FFmpeg filters that put every camera segment on the same canvas."""
    width = normalization.width
    height = normalization.height
    if normalization.mode == "scale":
        return f"scale={width}:{height},setsar=1"
    return (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1"
    )


def static_overlay_image_filter(
    overlay: StaticOverlay,
    input_index: int,
    video_width: int,
    video_height: int,
) -> str:
    """Return filters that prepare one transparent overlay image."""
    filters = [f"[{input_index}:v]format=rgba"]
    if overlay.width is not None or overlay.height is not None:
        width = overlay.width if overlay.width is not None else -1
        height = overlay.height if overlay.height is not None else -1
        filters.append(f"scale={width}:{height}")
    elif overlay.fit_to_video:
        filters.append(f"scale={video_width}:{video_height}")
    if overlay.opacity < 1:
        filters.append(f"colorchannelmixer=aa={overlay.opacity}")
    return ",".join(filters) + f"[ov{input_index}]"


def static_overlay_video_filter(
    base_label: str,
    overlay: StaticOverlay,
    input_index: int,
    output_label: str,
) -> str:
    """Return the FFmpeg overlay filter for one prepared static image."""
    return (
        f"{base_label}[ov{input_index}]"
        f"overlay=x={overlay.x}:y={overlay.y}:shortest=1:format=auto"
        f"[{output_label}]"
    )


def static_overlay_output_suffix(overlay_set: str | None) -> str:
    """Return the filename suffix used for static-overlaid clips."""
    if overlay_set is None:
        return STATIC_OVERLAY_SUFFIX
    return f"{STATIC_OVERLAY_SUFFIX}_{overlay_set}"


def static_overlay_output_dir(
    args: argparse.Namespace,
    camera_name: str,
    overlay_set: str | None,
) -> Path:
    """Return the output folder for one static overlay set."""
    root = args.output_dir / camera_name / "static_overlaid"
    if overlay_set is None:
        return root
    return root / overlay_set


def static_overlay_output_path(
    args: argparse.Namespace,
    camera_name: str,
    base: str,
    overlay_set: str | None,
) -> Path:
    """Return the static-overlaid clip path for one camera/chunk/set."""
    return (
        static_overlay_output_dir(args, camera_name, overlay_set)
        / f"{base}{static_overlay_output_suffix(overlay_set)}.mp4"
    )


def strip_static_overlay_suffix(stem: str) -> str:
    """Normalize a static-overlaid stem back to its source chunk stem."""
    match = re.fullmatch(
        rf"(.+){re.escape(STATIC_OVERLAY_SUFFIX)}(?:_[A-Za-z0-9_.-]+)?",
        stem,
    )
    return match.group(1) if match else stem


def base_chunk_name(stem: str) -> str:
    """Reduce a processed filename stem to its original small-chunk name."""
    stem = strip_static_overlay_suffix(stem)
    for suffix in ("_cleaned_overlay", "_overlay", "_cleaned"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def chunk_sort_key(base: str) -> tuple[int, int, int, str]:
    """Sort names such as big2_small3 in their natural numeric order."""
    match = re.fullmatch(r"big(\d+)_small(\d+)", base)
    if match:
        return (0, int(match.group(1)), int(match.group(2)), base)
    return (1, 0, 0, base)


def crop_image_array(image, crop: CropBox | None):
    """Crop an image-like array when it still appears to be in raw-video space."""
    if crop is None:
        return image
    height, width = image.shape[:2]
    if (width, height) == (crop.width, crop.height):
        return image
    if width >= crop.x + crop.width and height >= crop.y + crop.height:
        return image[crop.y : crop.y + crop.height, crop.x : crop.x + crop.width].copy()
    return image


def configure_data_paths(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Apply defaults, config file values, environment overrides, and CLI paths."""
    config_path = args.config
    if config_path is None:
        env_config = os.environ.get("SKATER_PIPELINE_CONFIG")
        config_path = Path(env_config) if env_config else DEFAULT_CONFIG_PATH

    if args.write_config:
        write_config_template(args.write_config)
        print(f"Wrote config template to: {args.write_config}")
        raise SystemExit(0)

    config = load_path_config(config_path)
    data_root_value = (
        args.data_root
        or os.environ.get("SKATER_DATA_ROOT")
        or config.get("data_root")
    )
    data_root = Path(data_root_value).expanduser() if data_root_value else None

    input_value = (
        args.input_dir
        or os.environ.get("SKATER_INPUT_DIR")
        or config.get("input_dir")
        or DEFAULT_INPUT_DIR
    )
    output_value = (
        args.output_dir
        or os.environ.get("SKATER_OUTPUT_DIR")
        or config.get("output_dir")
        or DEFAULT_OUTPUT_DIR
    )

    args.config = config_path
    args.data_root = data_root
    args.input_dir = resolve_data_path(Path(input_value).expanduser(), data_root)
    args.output_dir = resolve_data_path(Path(output_value).expanduser(), data_root)

    if args.input_dir == args.output_dir:
        parser.error("--input-dir and --output-dir must be different paths")


def show_configured_paths(args: argparse.Namespace) -> None:
    """Print the effective data paths after config/env/CLI resolution."""
    print(f"Config file: {args.config}")
    print(f"Data root: {args.data_root or '(none)'}")
    print(f"Input dir: {args.input_dir}")
    print(f"Output dir: {args.output_dir}")


def require_tool(name: str) -> None:
    """Fail early when a required command-line video tool is unavailable."""
    if shutil.which(name) is None:
        raise RuntimeError(f"Required command-line tool not found: {name}")


def run_command(cmd: list[str], dry_run: bool = False) -> None:
    """Run a command or print it when previewing a video-processing stage."""
    print(" ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def timecode_to_seconds(timecode: str, fps: int) -> float:
    """Convert HH:MM:SS or HH:MM:SS:FF timecodes to seconds."""
    parts = [int(part) for part in timecode.split(":")]
    if len(parts) == 3:
        h, m, s = parts
        frame = 0
    elif len(parts) == 4:
        h, m, s, frame = parts
    else:
        raise ValueError(f"Timecode must be HH:MM:SS or HH:MM:SS:FF: {timecode}")
    return h * 3600 + m * 60 + s + frame / fps


def best_small_chunk_duration(durations: list[float], lower: float, upper: float) -> float:
    """Choose a duration that leaves the least unused footage across chunks."""
    best_duration = lower
    best_remainder = math.inf
    step = 0.25
    n_steps = int((upper - lower) / step) + 1
    for i in range(n_steps):
        candidate = lower + i * step
        remainder = sum(duration % candidate for duration in durations)
        if remainder < best_remainder:
            best_remainder = remainder
            best_duration = candidate
    return best_duration


def camera_dirs(input_dir: Path) -> list[Path]:
    """Return camera folders in stable numeric/name order."""
    cameras_root = input_dir / "cameras"
    if not cameras_root.exists():
        raise FileNotFoundError(f"Missing camera folder: {cameras_root}")
    dirs = [p for p in cameras_root.iterdir() if p.is_dir()]
    if not dirs:
        raise FileNotFoundError(f"No camera folders found in: {cameras_root}")
    return sorted(dirs, key=lambda p: p.name)


def raw_video_path(camera_dir: Path) -> Path:
    """Find the raw recording for one camera folder."""
    preferred = camera_dir / "raw.mp4"
    if preferred.exists():
        return preferred
    videos = sorted(p for p in camera_dir.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS)
    if len(videos) != 1:
        raise FileNotFoundError(
            f"Expected raw.mp4 or exactly one video file in {camera_dir}; found {len(videos)}."
        )
    return videos[0]


def read_first_frame(camera_dir: Path) -> int:
    """Read the synchronization first-frame value for one camera."""
    path = camera_dir / "first_frame.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing synchronization file: {path}")
    return int(path.read_text(encoding="utf-8").strip())


def read_timing(input_dir: Path) -> list[dict[str, str]]:
    """Read chunk start/end timecodes from input_dir/timing.csv."""
    path = input_dir / "timing.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing timing file: {path}")
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    required = {"chunk", "start", "end"}
    if not rows or not required.issubset(rows[0].keys()):
        raise ValueError("timing.csv must contain columns: chunk,start,end")
    return rows


def synced_chunk_plan(
    input_dir: Path,
    fps: int,
    lead_in_seconds: float,
) -> list[list[dict[str, object]]]:
    """Calculate synchronized cut times for every camera and big chunk."""
    cameras = camera_dirs(input_dir)
    first_frames = [read_first_frame(camera_dir) for camera_dir in cameras]
    timing_rows = read_timing(input_dir)
    ref_first_frame = first_frames[0]

    plan = []
    for camera_index, (camera_dir, first_frame) in enumerate(
        zip(cameras, first_frames), start=1
    ):
        offset = (first_frame - ref_first_frame) / fps
        raw_video = raw_video_path(camera_dir)
        camera_plan = []
        for row in timing_rows:
            start = timecode_to_seconds(row["start"], fps) + lead_in_seconds + offset
            end = timecode_to_seconds(row["end"], fps) + offset
            camera_plan.append(
                {
                    "camera": camera_index,
                    "camera_name": camera_dir.name,
                    "chunk": int(row["chunk"]),
                    "video_file": raw_video,
                    "start": start,
                    "end": end,
                    "duration": end - start,
                }
            )
        plan.append(camera_plan)
    return plan


def get_video_info(video_path: Path) -> tuple[int, int, float]:
    """Return video width, height, and frame rate using FFprobe."""
    require_tool("ffprobe")
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate",
            "-of",
            "json",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    info = json.loads(result.stdout)["streams"][0]
    width = int(info["width"])
    height = int(info["height"])
    num, den = map(int, info["r_frame_rate"].split("/"))
    return width, height, num / den


def get_video_duration(path: Path) -> float:
    """Return video duration in seconds using FFprobe."""
    require_tool("ffprobe")
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=duration",
        "-of",
        "json",
        str(path),
    ]
    out = subprocess.check_output(cmd)
    return float(json.loads(out)["streams"][0]["duration"])


def camera_output_root(args: argparse.Namespace, camera_name: str) -> Path:
    """Return the root output folder for one camera."""
    return args.output_dir / camera_name


def big_chunk_dir(args: argparse.Namespace, camera_name: str) -> Path:
    """Return the folder for synchronized big chunks from one camera."""
    return camera_output_root(args, camera_name) / "big"


def small_chunk_dir(args: argparse.Namespace, camera_name: str) -> Path:
    """Return the folder for small chunks from one camera."""
    return camera_output_root(args, camera_name) / "small"


def processed_source_videos(
    args: argparse.Namespace,
    camera_name: str,
) -> list[Path]:
    """Prefer cleaned clips when present, otherwise use the original clips."""
    cleaned_dir = camera_output_root(args, camera_name) / "cleaned"
    videos = []
    for small_video in sorted(
        small_chunk_dir(args, camera_name).glob(args.chunk_pattern)
    ):
        cleaned_video = cleaned_dir / f"{small_video.stem}_cleaned.mp4"
        videos.append(cleaned_video if cleaned_video.exists() else small_video)
    return videos
