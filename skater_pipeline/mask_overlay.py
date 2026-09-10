"""Detect the skater and render coloured landmark overlays."""

import argparse
import csv
import subprocess
import time
from pathlib import Path

from .config import (
    DEFAULT_COLOR_MAP,
    PROGRESS_INTERVAL_SECONDS,
    format_duration,
    load_video_geometry,
    progress_summary,
)
from .video import (
    camera_dirs,
    crop_image_array,
    processed_source_videos,
    require_tool,
    small_chunk_dir,
)


def read_color_map(
    input_dir: Path,
) -> dict[tuple[int, int, int], tuple[tuple[int, int, int], float]]:
    """Load input_dir/color_map.csv when present, otherwise use defaults."""
    path = input_dir / "color_map.csv"
    if not path.exists():
        return DEFAULT_COLOR_MAP

    color_map = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mask = (int(row["mask_b"]), int(row["mask_g"]), int(row["mask_r"]))
            overlay = (
                int(row["overlay_b"]),
                int(row["overlay_g"]),
                int(row["overlay_r"]),
            )
            color_map[mask] = (overlay, float(row["alpha"]))
    return color_map


def create_mask_overlay(args: argparse.Namespace) -> None:
    """Generate moving masks and render coloured overlays in one video pass."""
    import cv2
    import numpy as np

    require_tool("ffmpeg")
    geometry = load_video_geometry(args.input_dir)
    color_map = read_color_map(args.input_dir)
    colors = np.array(list(color_map.keys()), dtype=np.int16)
    overlays = np.array([value[0] for value in color_map.values()], dtype=np.float32)
    alphas = np.array([value[1] for value in color_map.values()], dtype=np.float32)

    def align_image_to_frame(
        image: np.ndarray,
        frame: np.ndarray,
        interpolation: int,
    ) -> np.ndarray:
        """Resize a reference image once when its dimensions differ."""
        frame_h, frame_w = frame.shape[:2]
        if image.shape[:2] == (frame_h, frame_w):
            return image
        return cv2.resize(image, (frame_w, frame_h), interpolation=interpolation)

    processed_videos = 0
    stage_started_at = time.perf_counter()
    stage_frame_count = 0
    for camera_input_dir in camera_dirs(args.input_dir):
        camera_name = camera_input_dir.name

        empty_frame_path = camera_input_dir / "empty_frame.png"
        overlay_mask_path = camera_input_dir / "overlay_mask.png"
        if not empty_frame_path.exists() or not overlay_mask_path.exists():
            print(f"Skipping {camera_name}: no empty_frame.png/overlay_mask.png")
            continue

        ref = cv2.imread(str(empty_frame_path), cv2.IMREAD_COLOR)
        background_mask = cv2.imread(str(overlay_mask_path), cv2.IMREAD_UNCHANGED)
        if ref is None:
            raise RuntimeError(f"Cannot read empty frame: {empty_frame_path}")
        if background_mask is None:
            raise RuntimeError(f"Cannot read overlay mask: {overlay_mask_path}")
        if background_mask.ndim == 2:
            background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGRA)
        elif background_mask.shape[2] == 3:
            background_mask = cv2.cvtColor(background_mask, cv2.COLOR_BGR2BGRA)

        crop = geometry.crops.get(camera_name)
        ref = crop_image_array(ref, crop)
        background_mask = crop_image_array(background_mask, crop)

        source_videos = processed_source_videos(args, camera_name)
        if not source_videos:
            print(
                f"Skipping {camera_name}: no videos matching "
                f"{args.chunk_pattern!r} in {small_chunk_dir(args, camera_name)}"
            )
            continue

        output_dir = args.output_dir / camera_name / "overlaid"
        output_dir.mkdir(parents=True, exist_ok=True)

        for video_path in source_videos:
            output_path = output_dir / f"{video_path.stem}_overlay.mp4"
            video_started_at = time.perf_counter()
            print(
                f"Creating mask overlay: {video_path} -> {output_path}",
                flush=True,
            )
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {video_path}")
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(f"Cannot read video: {video_path}")

            h, w = frame.shape[:2]
            fps = cap.get(cv2.CAP_PROP_FPS) or args.fps
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_ref = align_image_to_frame(ref, frame, cv2.INTER_LINEAR)
            video_ref_f32 = video_ref.astype(np.float32)
            video_background_mask = align_image_to_frame(
                background_mask, frame, cv2.INTER_NEAREST
            )

            # The coloured overlay regions are static. Match them once instead
            # of constructing large colour-comparison arrays for every frame.
            static_mask_i16 = video_background_mask[:, :, :3].astype(np.int16)
            colour_diff = np.abs(
                static_mask_i16[:, :, None, :] - colors[None, None, :, :]
            )
            colour_matches = np.all(colour_diff < args.color_tolerance, axis=-1)
            static_idx = np.argmax(colour_matches, axis=-1)
            static_valid = np.any(colour_matches, axis=-1)
            static_overlay = overlays[static_idx]
            static_alpha = alphas[static_idx][..., None]
            static_one_minus_alpha = 1.0 - static_alpha
            static_overlay_term = static_overlay * static_alpha
            del colour_diff, colour_matches, static_mask_i16

            # White normally identifies the skater and is not in color_map.
            # Preserve the old behaviour if a custom map explicitly includes it.
            white_matches = np.all(
                np.abs(colors - np.array([255, 255, 255], dtype=np.int16))
                < args.color_tolerance,
                axis=-1,
            )
            white_valid = bool(np.any(white_matches))
            white_idx = int(np.argmax(white_matches)) if white_valid else 0
            close_kernel = np.ones(tuple(args.close_kernel), np.uint8)

            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{w}x{h}",
                "-r",
                str(int(fps)),
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
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_idx = 0
            render_started_at = time.perf_counter()
            last_progress_at = render_started_at
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Detect the moving skater and use the result immediately;
                # no per-frame mask image is written or read.
                diff = cv2.absdiff(frame.astype("float32"), video_ref_f32)
                diff = cv2.GaussianBlur(diff, tuple(args.blur_kernel), 0)
                diff_gray = cv2.cvtColor(diff.astype(np.uint8), cv2.COLOR_BGR2GRAY)
                _, skater_mask = cv2.threshold(
                    diff_gray, args.mask_threshold, 255, cv2.THRESH_BINARY
                )
                skater_mask = cv2.morphologyEx(
                    skater_mask, cv2.MORPH_CLOSE, close_kernel
                )
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                    skater_mask, connectivity=8
                )
                if num_labels > 1:
                    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                    skater_mask = labels == largest_label
                else:
                    skater_mask = np.zeros((h, w), dtype=bool)

                frame_f32 = frame.astype(np.float32)
                if white_valid:
                    frame_idx_map = static_idx.copy()
                    frame_idx_map[skater_mask] = white_idx
                    overlay_map = overlays[frame_idx_map]
                    alpha_map = alphas[frame_idx_map][..., None]
                    valid = static_valid.copy()
                    valid[skater_mask] = True
                    result = np.where(
                        valid[..., None],
                        frame_f32 * (1.0 - alpha_map) + overlay_map * alpha_map,
                        frame_f32,
                    )
                else:
                    valid = static_valid & ~skater_mask
                    result = np.where(
                        valid[..., None],
                        frame_f32 * static_one_minus_alpha + static_overlay_term,
                        frame_f32,
                    )
                proc.stdin.write(np.clip(result, 0, 255).astype(np.uint8).tobytes())
                frame_idx += 1
                now = time.perf_counter()
                if now - last_progress_at >= PROGRESS_INTERVAL_SECONDS:
                    print(
                        progress_summary(
                            "  Mask overlay progress",
                            frame_idx,
                            total_frames,
                            render_started_at,
                        ),
                        flush=True,
                    )
                    last_progress_at = now

            if proc.stdin:
                proc.stdin.close()
            proc.wait()
            cap.release()
            render_elapsed = time.perf_counter() - render_started_at
            video_elapsed = time.perf_counter() - video_started_at
            render_rate = frame_idx / render_elapsed if render_elapsed > 0 else 0.0
            total_rate = frame_idx / video_elapsed if video_elapsed > 0 else 0.0
            print(
                f"Finished mask overlay: {frame_idx} frames in "
                f"{format_duration(render_elapsed)} ({render_rate:.2f} frames/s), "
                f"total {format_duration(video_elapsed)} "
                f"({total_rate:.2f} frames/s) -> {output_path}",
                flush=True,
            )
            stage_frame_count += frame_idx
            processed_videos += 1

    if processed_videos == 0:
        raise FileNotFoundError(
            "No mask overlay inputs were processed. Check the input camera "
            "references and small video folders for the requested chunk pattern."
        )

    stage_elapsed = time.perf_counter() - stage_started_at
    stage_rate = stage_frame_count / stage_elapsed if stage_elapsed > 0 else 0.0
    print(
        f"Mask overlay stage complete: {processed_videos} videos, "
        f"{stage_frame_count} frames "
        f"in {format_duration(stage_elapsed)} ({stage_rate:.2f} frames/s)",
        flush=True,
    )
