"""Stage that composites configured transparent PNGs onto camera clips."""

import argparse
import fnmatch
from pathlib import Path

from .config import (
    STATIC_OVERLAY_CONFIG_FILENAME,
    VIDEO_EXTENSIONS,
    StaticOverlay,
    load_static_overlay_config,
    static_overlay_set_dir,
)
from .video import (
    base_chunk_name,
    camera_dirs,
    chunk_sort_key,
    get_video_info,
    require_tool,
    run_command,
    small_chunk_dir,
    static_overlay_image_filter,
    static_overlay_output_dir,
    static_overlay_output_path,
    static_overlay_video_filter,
)

def static_overlay_applies(
    overlay: StaticOverlay,
    camera_name: str,
    base: str,
) -> bool:
    """Return whether a configured static image applies to one camera/chunk."""
    if overlay.cameras is not None and camera_name not in overlay.cameras:
        return False
    if overlay.chunks is None:
        return True
    return any(
        fnmatch.fnmatch(base, pattern)
        or fnmatch.fnmatch(f"{base}.mp4", pattern)
        for pattern in overlay.chunks
    )


def static_overlay_source_candidates(
    args: argparse.Namespace,
    camera_name: str,
    base: str,
) -> list[Path]:
    """Return source candidates before optional static overlays are applied."""
    return [
        args.output_dir / camera_name / "cleaned" / f"{base}_cleaned.mp4",
        small_chunk_dir(args, camera_name) / f"{base}.mp4",
    ]


def static_overlay_source(args: argparse.Namespace, camera_name: str, base: str) -> Path:
    """Select the best available source clip for static image overlays."""
    for path in static_overlay_source_candidates(args, camera_name, base):
        if path.exists():
            return path
    raise FileNotFoundError(f"No static overlay source found for {camera_name}, {base}")


def static_overlay_bases_for_camera(args: argparse.Namespace, camera_name: str) -> set[str]:
    """Return source clip stems available for static image overlays."""
    bases = set()
    source_dirs = [
        args.output_dir / camera_name / "cleaned",
        small_chunk_dir(args, camera_name),
    ]
    for source_dir in source_dirs:
        if not source_dir.exists():
            continue
        for path in source_dir.iterdir():
            if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            base = base_chunk_name(path.stem)
            if fnmatch.fnmatch(f"{base}.mp4", args.chunk_pattern):
                bases.add(base)
    return bases


def render_static_overlays(args: argparse.Namespace) -> None:
    """Render configured transparent static images over each matching video."""
    config = load_static_overlay_config(args.input_dir, args.static_overlay_set)
    if config.path is None:
        if args.static_overlay_set is None:
            expected_path = args.input_dir / STATIC_OVERLAY_CONFIG_FILENAME
        else:
            expected_path = (
                static_overlay_set_dir(args.input_dir, args.static_overlay_set)
                / STATIC_OVERLAY_CONFIG_FILENAME
            )
        print(f"Skipping static_overlays: no {expected_path}")
        return
    if not config.overlays:
        print(f"Skipping static_overlays: no enabled overlays in {config.path}")
        return

    require_tool("ffmpeg")
    require_tool("ffprobe")

    processed_videos = 0
    for camera_input_dir in camera_dirs(args.input_dir):
        camera_name = camera_input_dir.name
        bases = sorted(
            static_overlay_bases_for_camera(args, camera_name),
            key=chunk_sort_key,
        )
        if not bases:
            print(f"Skipping {camera_name}: no static overlay source videos")
            continue

        output_dir = static_overlay_output_dir(
            args,
            camera_name,
            config.set_name,
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        for base in bases:
            matching_overlays = [
                overlay
                for overlay in config.overlays
                if static_overlay_applies(overlay, camera_name, base)
            ]
            if not matching_overlays:
                continue

            missing_images = [
                overlay.image
                for overlay in matching_overlays
                if not overlay.image.exists()
            ]
            if missing_images:
                missing = ", ".join(str(path) for path in missing_images)
                raise FileNotFoundError(f"Missing static overlay image(s): {missing}")

            video_path = static_overlay_source(args, camera_name, base)
            video_width, video_height, fps = get_video_info(video_path)
            output_path = static_overlay_output_path(
                args,
                camera_name,
                base,
                config.set_name,
            )
            print(f"Rendering static overlays: {video_path} -> {output_path}")

            cmd = ["ffmpeg", "-y", "-i", str(video_path)]
            for overlay in matching_overlays:
                cmd.extend(["-loop", "1", "-i", str(overlay.image)])

            filters = []
            current_label = "[0:v]"
            for input_index, overlay in enumerate(matching_overlays, start=1):
                filters.append(
                    static_overlay_image_filter(
                        overlay,
                        input_index,
                        video_width,
                        video_height,
                    )
                )
                output_label = f"static{input_index}"
                filters.append(
                    static_overlay_video_filter(
                        current_label,
                        overlay,
                        input_index,
                        output_label,
                    )
                )
                current_label = f"[{output_label}]"

            cmd.extend(
                [
                    "-filter_complex",
                    ";\n".join(filters),
                    "-map",
                    current_label,
                    "-r",
                    str(fps),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "18",
                    "-pix_fmt",
                    "yuv420p",
                    "-an",
                    str(output_path),
                ]
            )
            run_command(cmd, dry_run=args.dry_run)
            processed_videos += 1

    if processed_videos == 0:
        raise FileNotFoundError(
            "No static overlay videos were processed. Check that configured "
            "camera/chunk filters match available source clips under the output folder."
        )
