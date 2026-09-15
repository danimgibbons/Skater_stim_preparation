"""Composite configured PNG overlays, optionally behind the moving skater."""

import argparse
import fnmatch
import subprocess
import time
from pathlib import Path

from .config import (
    OVERLAY_CONFIG_FILENAME,
    PROGRESS_INTERVAL_SECONDS,
    VIDEO_EXTENSIONS,
    Overlay,
    available_overlay_sets,
    format_duration,
    load_overlay_config,
    load_video_geometry,
    overlay_set_dir,
    progress_summary,
)
from .video import (
    base_chunk_name,
    camera_dirs,
    chunk_sort_key,
    crop_image_array,
    get_video_info,
    overlay_image_filter,
    overlay_output_dir,
    overlay_output_path,
    overlay_video_filter,
    require_tool,
    run_command,
    small_chunk_dir,
)


def overlay_applies(overlay: Overlay, camera_name: str, base: str) -> bool:
    """Return whether a configured PNG applies to one camera/chunk."""
    if overlay.cameras is not None and camera_name not in overlay.cameras:
        return False
    if overlay.chunks is None:
        return True
    return any(
        fnmatch.fnmatch(base, pattern)
        or fnmatch.fnmatch(f"{base}.mp4", pattern)
        for pattern in overlay.chunks
    )


def overlay_source_candidates(
    args: argparse.Namespace,
    camera_name: str,
    base: str,
) -> list[Path]:
    """Return source candidates before PNG overlays are applied."""
    return [
        args.output_dir / camera_name / "cleaned" / f"{base}_cleaned.mp4",
        small_chunk_dir(args, camera_name) / f"{base}.mp4",
    ]


def overlay_source(args: argparse.Namespace, camera_name: str, base: str) -> Path:
    """Select the best available source clip for PNG overlays."""
    for path in overlay_source_candidates(args, camera_name, base):
        if path.exists():
            return path
    raise FileNotFoundError(f"No overlay source found for {camera_name}, {base}")


def overlay_bases_for_camera(args: argparse.Namespace, camera_name: str) -> set[str]:
    """Return source clip stems available for PNG overlays."""
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


def build_unmasked_overlay_filters(
    overlays: list[Overlay],
    video_width: int,
    video_height: int,
) -> tuple[list[str], str]:
    """Build the direct FFmpeg graph used when no skater mask is requested."""
    filters = []
    current_label = "[0:v]"
    for input_index, overlay in enumerate(overlays, start=1):
        filters.append(
            overlay_image_filter(
                overlay,
                input_index,
                video_width,
                video_height,
            )
        )
        output_label = f"overlay{input_index}"
        filters.append(
            overlay_video_filter(
                current_label,
                overlay,
                input_index,
                output_label,
            )
        )
        current_label = f"[{output_label}]"
    return filters, current_label


def render_unmasked_overlays(
    args: argparse.Namespace,
    video_path: Path,
    output_path: Path,
    overlays: list[Overlay],
    video_width: int,
    video_height: int,
    fps: float,
) -> None:
    """Use FFmpeg directly when every PNG is allowed to cover the skater."""
    started_at = time.perf_counter()
    cmd = ["ffmpeg", "-y", "-i", str(video_path)]
    for overlay in overlays:
        cmd.extend(["-loop", "1", "-i", str(overlay.image)])
    filters, output_label = build_unmasked_overlay_filters(
        overlays,
        video_width,
        video_height,
    )
    cmd.extend(
        [
            "-filter_complex",
            ";\n".join(filters),
            "-map",
            output_label,
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
    if not args.dry_run:
        print(
            f"Finished overlays in "
            f"{format_duration(time.perf_counter() - started_at)} -> {output_path}"
        )


def prepare_overlay_canvas(
    overlay: Overlay,
    video_width: int,
    video_height: int,
):
    """Render one positioned PNG to a full-frame BGRA array once per video."""
    import numpy as np

    image_filter = overlay_image_filter(
        overlay,
        1,
        video_width,
        video_height,
    )
    video_filter = overlay_video_filter(
        "[0:v]",
        overlay,
        1,
        "prepared",
    )
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"color=c=black@0.0:s={video_width}x{video_height}:r=1,format=rgba",
        "-i",
        str(overlay.image),
        "-filter_complex",
        f"{image_filter};{video_filter}",
        "-map",
        "[prepared]",
        "-frames:v",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgra",
        "-",
    ]
    result = subprocess.run(cmd, capture_output=True, check=True)
    expected_size = video_width * video_height * 4
    if len(result.stdout) != expected_size:
        raise RuntimeError(
            f"Could not prepare overlay PNG at video size: {overlay.image}"
        )
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(
        (video_height, video_width, 4)
    )


def overlay_affine_maps(canvases, included: list[bool]):
    """Precombine static PNG layers as result = frame * multiplier + addition."""
    import numpy as np

    height, width = canvases[0].shape[:2]
    multiplier = np.ones((height, width, 1), dtype=np.float32)
    addition = np.zeros((height, width, 3), dtype=np.float32)
    for canvas, include in zip(canvases, included):
        if not include:
            continue
        alpha = canvas[:, :, 3:4].astype(np.float32) / 255.0
        one_minus_alpha = 1.0 - alpha
        addition *= one_minus_alpha
        addition += canvas[:, :, :3].astype(np.float32) * alpha
        multiplier *= one_minus_alpha
    return multiplier, addition


def render_fused_masked_overlays(
    args: argparse.Namespace,
    video_path: Path,
    output_path: Path,
    overlays: list[Overlay],
    empty_frame,
) -> None:
    """Detect and mask the skater while rendering PNGs in one video pass."""
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError(f"Cannot read video: {video_path}")

    height, width = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS) or args.fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if empty_frame.shape[:2] == (height, width):
        video_reference = empty_frame
    else:
        video_reference = cv2.resize(
            empty_frame,
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )
    video_reference_f32 = video_reference.astype(np.float32)
    close_kernel = np.ones(tuple(args.close_kernel), np.uint8)

    canvases = [
        prepare_overlay_canvas(overlay, width, height) for overlay in overlays
    ]
    all_multiplier, all_addition = overlay_affine_maps(
        canvases,
        [True] * len(overlays),
    )
    visible_multiplier, visible_addition = overlay_affine_maps(
        canvases,
        [not overlay.mask_skater for overlay in overlays],
    )

    ffmpeg_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    if proc.stdin is None:
        cap.release()
        raise RuntimeError("Could not open the overlay FFmpeg pipe")

    started_at = time.perf_counter()
    last_progress_at = started_at
    frame_count = 0
    while ret:
        frame_f32 = frame.astype(np.float32)
        diff = cv2.absdiff(frame_f32, video_reference_f32)
        diff = cv2.GaussianBlur(diff, tuple(args.blur_kernel), 0)
        diff_gray = cv2.cvtColor(diff.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        _, candidate_mask = cv2.threshold(
            diff_gray,
            args.mask_threshold,
            255,
            cv2.THRESH_BINARY,
        )
        candidate_mask = cv2.morphologyEx(
            candidate_mask,
            cv2.MORPH_CLOSE,
            close_kernel,
        )
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            candidate_mask,
            connectivity=8,
        )
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            skater_mask = labels == largest_label
        else:
            skater_mask = np.zeros((height, width), dtype=bool)

        result = frame_f32 * all_multiplier + all_addition
        if np.any(skater_mask):
            result[skater_mask] = (
                frame_f32[skater_mask] * visible_multiplier[skater_mask]
                + visible_addition[skater_mask]
            )
        proc.stdin.write(np.clip(result, 0, 255).astype(np.uint8).tobytes())

        frame_count += 1
        now = time.perf_counter()
        if now - last_progress_at >= PROGRESS_INTERVAL_SECONDS:
            print(
                progress_summary(
                    "  Fused overlay progress",
                    frame_count,
                    total_frames,
                    started_at,
                ),
                flush=True,
            )
            last_progress_at = now
        ret, frame = cap.read()

    cap.release()
    proc.stdin.close()
    returncode = proc.wait()
    if returncode != 0:
        raise RuntimeError(f"FFmpeg failed to write overlay: {output_path}")
    elapsed = time.perf_counter() - started_at
    rate = frame_count / elapsed if elapsed > 0 else 0.0
    print(
        f"Finished fused overlay: {frame_count} frames in "
        f"{format_duration(elapsed)} ({rate:.2f} frames/s) -> {output_path}",
        flush=True,
    )


def render_overlays(args: argparse.Namespace) -> None:
    """Render configured PNGs, optionally masking the skater in one pass."""
    config = load_overlay_config(args.input_dir, args.overlay_set)
    if config.path is None:
        expected_path = (
            args.input_dir / OVERLAY_CONFIG_FILENAME
            if args.overlay_set is None
            else overlay_set_dir(args.input_dir, args.overlay_set)
            / OVERLAY_CONFIG_FILENAME
        )
        if args.overlay_set is not None:
            available = available_overlay_sets(args.input_dir)
            choices = ", ".join(available) if available else "(none)"
            raise FileNotFoundError(
                f"Overlay set {args.overlay_set!r} has no config at "
                f"{expected_path}. Available sets: {choices}."
            )
        print(f"Skipping overlays: no {expected_path}")
        return
    if not config.overlays:
        print(f"Skipping overlays: no enabled overlays in {config.path}")
        return

    require_tool("ffmpeg")
    require_tool("ffprobe")
    geometry = load_video_geometry(args.input_dir)
    processed_videos = 0

    for camera_input_dir in camera_dirs(args.input_dir):
        camera_name = camera_input_dir.name
        bases = sorted(
            overlay_bases_for_camera(args, camera_name),
            key=chunk_sort_key,
        )
        if not bases:
            print(f"Skipping {camera_name}: no overlay source videos")
            continue

        output_dir = overlay_output_dir(args, camera_name, config.set_name)
        output_dir.mkdir(parents=True, exist_ok=True)
        empty_frame = None

        for base in bases:
            matching_overlays = [
                overlay
                for overlay in config.overlays
                if overlay_applies(overlay, camera_name, base)
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
                raise FileNotFoundError(f"Missing overlay PNG(s): {missing}")

            video_path = overlay_source(args, camera_name, base)
            video_width, video_height, fps = get_video_info(video_path)
            output_path = overlay_output_path(
                args,
                camera_name,
                base,
                config.set_name,
            )
            uses_skater_mask = any(
                overlay.mask_skater for overlay in matching_overlays
            )
            print(f"Rendering overlays: {video_path} -> {output_path}")
            if not uses_skater_mask:
                render_unmasked_overlays(
                    args,
                    video_path,
                    output_path,
                    matching_overlays,
                    video_width,
                    video_height,
                    fps,
                )
                processed_videos += 1
                continue
            if args.dry_run:
                print(
                    "Would detect the skater and render masked PNG overlays "
                    "in one streaming pass"
                )
                processed_videos += 1
                continue

            if empty_frame is None:
                import cv2

                empty_frame_path = camera_input_dir / "empty_frame.png"
                empty_frame = cv2.imread(str(empty_frame_path), cv2.IMREAD_COLOR)
                if empty_frame is None:
                    raise FileNotFoundError(
                        f"Masked overlays require an empty frame: {empty_frame_path}"
                    )
                empty_frame = crop_image_array(
                    empty_frame,
                    geometry.crops.get(camera_name),
                )
            render_fused_masked_overlays(
                args,
                video_path,
                output_path,
                matching_overlays,
                empty_frame,
            )
            processed_videos += 1

    if processed_videos == 0:
        raise FileNotFoundError(
            "No overlay videos were processed. Check that configured camera/chunk "
            "filters match available source clips under the output folder."
        )
