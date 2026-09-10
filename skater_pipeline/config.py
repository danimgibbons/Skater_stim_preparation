"""Prepare multi-camera skating videos for MEG stimulus presentation.

The pipeline uses a fixed input-folder structure and can run each editing stage
independently so intermediate videos can be inspected before continuing.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path


STAGES = [
    "cut_big",
    "cut_small",
    "clean_bg",
    "export_masks",
    "mask_overlay",
    "static_overlays",
    "camera_cut",
]
# Frame-by-frame mask export is diagnostic and not part of a normal run.
DEFAULT_STAGES = [stage for stage in STAGES if stage != "export_masks"]
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v"}
DEFAULT_COLOR_MAP = {
    (255, 0, 0): ((255, 0, 0), 0.2),
    (0, 0, 255): ((0, 0, 255), 0.2),
    (100, 0, 0): ((255, 0, 0), 0.4),
    (0, 100, 0): ((0, 255, 0), 0.4),
    (0, 0, 100): ((0, 0, 255), 0.4),
    (0, 0, 0): ((0, 255, 255), 0.4),
}
DEFAULT_INPUT_DIR = Path("Vids/input")
DEFAULT_OUTPUT_DIR = Path("Vids/output")
DEFAULT_CONFIG_PATH = Path("pipeline_config.json")
GEOMETRY_CONFIG_FILENAME = "video_geometry.json"
STATIC_OVERLAY_CONFIG_FILENAME = "static_overlays.json"
STATIC_OVERLAY_ROOT_DIR = "static_overlays"
STATIC_OVERLAY_ROOT_DIR_ALIASES = ("static_overlays", "static overlays")
STATIC_OVERLAY_SUFFIX = "_static_overlay"
CONFIG_TEMPLATE = {
    "data_root": "/path/to/mounted/secure/server/skater-data",
    "input_dir": "input",
    "output_dir": "output",
}
PROGRESS_INTERVAL_SECONDS = 5.0


def format_duration(seconds: float) -> str:
    """Format an elapsed duration for concise terminal progress output."""
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def progress_summary(
    label: str,
    completed: int,
    total: int,
    started_at: float,
) -> str:
    """Build a progress line containing elapsed time, throughput, and ETA."""
    elapsed = time.perf_counter() - started_at
    rate = completed / elapsed if elapsed > 0 else 0.0
    total_text = str(total) if total > 0 else "?"
    summary = (
        f"{label}: {completed}/{total_text} frames, {rate:.2f} frames/s, "
        f"elapsed {format_duration(elapsed)}"
    )
    if total > completed and rate > 0:
        summary += f", ETA {format_duration((total - completed) / rate)}"
    return summary


@dataclass(frozen=True)
class CropBox:
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class Normalization:
    width: int
    height: int
    mode: str


@dataclass(frozen=True)
class VideoGeometry:
    crops: dict[str, CropBox]
    normalization: Normalization | None
    path: Path | None


@dataclass(frozen=True)
class StaticOverlay:
    image: Path
    x: str
    y: str
    width: int | None
    height: int | None
    fit_to_video: bool
    opacity: float
    cameras: list[str] | None
    chunks: list[str] | None


@dataclass(frozen=True)
class StaticOverlayConfig:
    overlays: list[StaticOverlay]
    path: Path | None
    set_name: str | None


def resolve_data_path(path: Path, data_root: Path | None) -> Path:
    """Resolve a configured data path, optionally relative to a shared data root."""
    return path if path.is_absolute() or data_root is None else data_root / path


def load_path_config(path: Path | None) -> dict[str, str]:
    """Read optional local path configuration from JSON."""
    if path is None:
        return {}
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a JSON object: {path}")
    allowed_keys = {"data_root", "input_dir", "output_dir"}
    unknown_keys = sorted(set(config) - allowed_keys)
    if unknown_keys:
        raise ValueError(
            f"Unknown config key(s) in {path}: {', '.join(unknown_keys)}. "
            f"Allowed keys: {', '.join(sorted(allowed_keys))}."
        )
    return {key: str(value) for key, value in config.items() if value not in (None, "")}


def write_config_template(path: Path) -> None:
    """Create a local path configuration template."""
    if path.exists():
        raise FileExistsError(f"Config file already exists: {path}")
    path.write_text(json.dumps(CONFIG_TEMPLATE, indent=2) + "\n", encoding="utf-8")


def parse_pixel_int(value: object, field: str) -> int:
    """Parse one JSON pixel value and keep accidental fractional crops explicit."""
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer pixel value")
    if isinstance(value, str):
        value = value.replace(",", ".")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer pixel value") from exc
    if not number.is_integer():
        raise ValueError(f"{field} must be an integer pixel value, got {value!r}")
    return int(number)


def parse_even_dimension(value: object, field: str) -> int:
    """Parse a positive even video dimension for H.264/yuv420p output."""
    dimension = parse_pixel_int(value, field)
    if dimension <= 0:
        raise ValueError(f"{field} must be greater than zero")
    if dimension % 2:
        raise ValueError(f"{field} must be even for H.264/yuv420p output")
    return dimension


def parse_positive_int(value: object, field: str) -> int:
    """Parse a positive integer pixel value."""
    number = parse_pixel_int(value, field)
    if number <= 0:
        raise ValueError(f"{field} must be greater than zero")
    return number


def parse_filter_expression(value: object, field: str, default: str) -> str:
    """Parse an FFmpeg position expression from JSON."""
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a number or FFmpeg expression string")
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise ValueError(f"{field} must be a number or FFmpeg expression string")


def parse_optional_overlay_dimension(value: object, field: str) -> int | None:
    """Parse optional static overlay scaling dimensions."""
    if value in (None, "auto"):
        return None
    return parse_positive_int(value, field)


def parse_optional_string_list(value: object, field: str) -> list[str] | None:
    """Parse an optional JSON string or list of strings."""
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a string or a list of strings")
    items = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item:
            raise ValueError(f"{field}[{index}] must be a non-empty string")
        items.append(item)
    return items


def parse_bool(value: object, field: str, default: bool = False) -> bool:
    """Parse an optional boolean JSON value."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise ValueError(f"{field} must be true or false")


def parse_crop_box(value: object, camera_name: str) -> CropBox:
    """Parse one camera crop rectangle from video_geometry.json."""
    if not isinstance(value, dict):
        raise ValueError(f"Crop for {camera_name} must be a JSON object")
    field_prefix = f"{camera_name}.crop"
    required = {"x", "y", "width", "height"}
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(
            f"Crop for {camera_name} is missing: {', '.join(missing)}"
        )
    x = parse_pixel_int(value["x"], f"{field_prefix}.x")
    y = parse_pixel_int(value["y"], f"{field_prefix}.y")
    width = parse_even_dimension(value["width"], f"{field_prefix}.width")
    height = parse_even_dimension(value["height"], f"{field_prefix}.height")
    if x < 0 or y < 0:
        raise ValueError(f"Crop for {camera_name} must use non-negative x/y")
    return CropBox(x=x, y=y, width=width, height=height)


def parse_normalization(value: object) -> Normalization | None:
    """Parse final output normalization settings from video_geometry.json."""
    if value in (None, False):
        return None
    if not isinstance(value, dict):
        raise ValueError("normalization must be a JSON object")
    required = {"width", "height"}
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"normalization is missing: {', '.join(missing)}")
    width = parse_even_dimension(value["width"], "normalization.width")
    height = parse_even_dimension(value["height"], "normalization.height")
    mode = str(value.get("mode", "scale"))
    if mode not in {"scale", "pad"}:
        raise ValueError("normalization.mode must be either 'scale' or 'pad'")
    return Normalization(width=width, height=height, mode=mode)


def load_video_geometry(input_dir: Path) -> VideoGeometry:
    """Load optional camera crop and final normalization settings."""
    path = input_dir / GEOMETRY_CONFIG_FILENAME
    if not path.exists():
        return VideoGeometry(crops={}, normalization=None, path=None)

    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError(f"{GEOMETRY_CONFIG_FILENAME} must contain a JSON object")

    cameras = config.get("cameras", {})
    if not isinstance(cameras, dict):
        raise ValueError("video_geometry.json field 'cameras' must be a JSON object")

    crops = {}
    for camera_name, camera_config in cameras.items():
        if not isinstance(camera_config, dict):
            raise ValueError(f"Geometry for {camera_name} must be a JSON object")
        crop_config = camera_config.get("crop")
        if crop_config is not None:
            crops[str(camera_name)] = parse_crop_box(crop_config, str(camera_name))

    normalization = parse_normalization(config.get("normalization"))
    return VideoGeometry(crops=crops, normalization=normalization, path=path)


def parse_static_overlay(
    value: object,
    image_root: Path,
    index: int,
) -> StaticOverlay:
    """Parse one transparent image overlay from static_overlays.json."""
    if not isinstance(value, dict):
        raise ValueError(f"overlays[{index}] must be a JSON object")
    field_prefix = f"overlays[{index}]"
    image_value = value.get("image")
    if not isinstance(image_value, str) or not image_value:
        raise ValueError(f"{field_prefix}.image must be a non-empty string")
    image = Path(image_value).expanduser()
    if not image.is_absolute():
        image = image_root / image

    opacity = float(value.get("opacity", 1.0))
    if opacity < 0 or opacity > 1:
        raise ValueError(f"{field_prefix}.opacity must be between 0 and 1")

    return StaticOverlay(
        image=image,
        x=parse_filter_expression(value.get("x"), f"{field_prefix}.x", "0"),
        y=parse_filter_expression(value.get("y"), f"{field_prefix}.y", "0"),
        width=parse_optional_overlay_dimension(
            value.get("width"),
            f"{field_prefix}.width",
        ),
        height=parse_optional_overlay_dimension(
            value.get("height"),
            f"{field_prefix}.height",
        ),
        fit_to_video=parse_bool(
            value.get("fit_to_video"),
            f"{field_prefix}.fit_to_video",
        ),
        opacity=opacity,
        cameras=parse_optional_string_list(value.get("cameras"), f"{field_prefix}.cameras"),
        chunks=parse_optional_string_list(value.get("chunks"), f"{field_prefix}.chunks"),
    )


def static_overlay_config_location(
    input_dir: Path,
    overlay_set: str | None,
) -> tuple[Path, Path]:
    """Return the config path and relative-image root for a static overlay set."""
    if overlay_set is None:
        return input_dir / STATIC_OVERLAY_CONFIG_FILENAME, input_dir
    config_dir = static_overlay_set_dir(input_dir, overlay_set)
    return config_dir / STATIC_OVERLAY_CONFIG_FILENAME, config_dir


def static_overlay_set_dir(input_dir: Path, overlay_set: str) -> Path:
    """Return the folder for a named static overlay set."""
    for root_dir in STATIC_OVERLAY_ROOT_DIR_ALIASES:
        candidate = input_dir / root_dir / overlay_set
        if candidate.exists():
            return candidate
    return input_dir / STATIC_OVERLAY_ROOT_DIR / overlay_set


def load_static_overlay_config(
    input_dir: Path,
    overlay_set: str | None = None,
) -> StaticOverlayConfig:
    """Load optional transparent static overlay settings."""
    path, image_root = static_overlay_config_location(input_dir, overlay_set)
    if not path.exists():
        return StaticOverlayConfig(overlays=[], path=None, set_name=overlay_set)

    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError(f"{STATIC_OVERLAY_CONFIG_FILENAME} must contain a JSON object")
    if config.get("enabled", True) is False:
        return StaticOverlayConfig(overlays=[], path=path, set_name=overlay_set)

    overlays_config = config.get("overlays", [])
    if not isinstance(overlays_config, list):
        raise ValueError("static_overlays.json field 'overlays' must be a list")

    overlays = [
        parse_static_overlay(overlay_config, image_root, index)
        for index, overlay_config in enumerate(overlays_config)
    ]
    return StaticOverlayConfig(overlays=overlays, path=path, set_name=overlay_set)
