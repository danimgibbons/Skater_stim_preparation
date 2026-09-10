"""Generate optional diagnostic masks for each video frame."""

import argparse
import time

from .config import (
    PROGRESS_INTERVAL_SECONDS,
    format_duration,
    load_video_geometry,
    progress_summary,
)
from .video import camera_dirs, crop_image_array, processed_source_videos


def export_frame_masks(args: argparse.Namespace) -> None:
    """Create one mask image per video frame using reference-frame differencing."""
    import cv2
    import numpy as np
    geometry = load_video_geometry(args.input_dir)

    def align_image_to_frame(
        image: np.ndarray,
        frame: np.ndarray,
        interpolation: int,
    ) -> np.ndarray:
        """Resize an image to match the current video frame when needed."""
        frame_h, frame_w = frame.shape[:2]
        if image.shape[:2] == (frame_h, frame_w):
            return image
        return cv2.resize(image, (frame_w, frame_h), interpolation=interpolation)

    processed_videos = 0
    stage_started_at = time.perf_counter()
    stage_frame_count = 0
    for camera_input_dir in camera_dirs(args.input_dir):
        empty_frame_path = camera_input_dir / "empty_frame.png"
        overlay_mask_path = camera_input_dir / "overlay_mask.png"
        if not empty_frame_path.exists() or not overlay_mask_path.exists():
            print(f"Skipping {camera_input_dir.name}: no empty_frame.png/overlay_mask.png")
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

        crop = geometry.crops.get(camera_input_dir.name)
        ref = crop_image_array(ref, crop)
        background_mask = crop_image_array(background_mask, crop)

        source_videos = processed_source_videos(args, camera_input_dir.name)
        if not source_videos:
            print(
                f"Skipping {camera_input_dir.name}: no videos matching "
                f"{args.chunk_pattern!r} in {small_chunk_dir(args, camera_input_dir.name)}"
            )
            continue

        for video_path in source_videos:
            output_dir = args.output_dir / "masks" / camera_input_dir.name / video_path.stem
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Exporting masks: {video_path} -> {output_dir}", flush=True)
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {video_path}")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_started_at = time.perf_counter()
            last_progress_at = video_started_at
            video_ref = None
            video_background_mask = None
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if video_ref is None:
                    video_ref = align_image_to_frame(ref, frame, cv2.INTER_LINEAR)
                    video_background_mask = align_image_to_frame(
                        background_mask, frame, cv2.INTER_NEAREST
                    )

                diff = cv2.absdiff(frame.astype("float32"), video_ref.astype("float32"))
                diff = cv2.GaussianBlur(diff, tuple(args.blur_kernel), 0)
                diff_gray = cv2.cvtColor(diff.astype(np.uint8), cv2.COLOR_BGR2GRAY)
                _, skater_mask = cv2.threshold(
                    diff_gray, args.mask_threshold, 255, cv2.THRESH_BINARY
                )
                kernel = np.ones(tuple(args.close_kernel), np.uint8)
                skater_mask = cv2.morphologyEx(skater_mask, cv2.MORPH_CLOSE, kernel)
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                    skater_mask, connectivity=8
                )

                if num_labels > 1:
                    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                    largest_blob = np.zeros_like(skater_mask)
                    largest_blob[labels == largest_label] = 255
                    skater_mask = largest_blob

                final_mask = video_background_mask.copy()
                final_mask[skater_mask == 255] = [255, 255, 255, 255]
                cv2.imwrite(str(output_dir / f"frame_{frame_idx:06d}.png"), final_mask)
                frame_idx += 1
                now = time.perf_counter()
                if now - last_progress_at >= PROGRESS_INTERVAL_SECONDS:
                    print(
                        progress_summary(
                            "  Mask progress", frame_idx, total_frames, video_started_at
                        ),
                        flush=True,
                    )
                    last_progress_at = now

            cap.release()
            video_elapsed = time.perf_counter() - video_started_at
            video_rate = frame_idx / video_elapsed if video_elapsed > 0 else 0.0
            print(
                f"Finished mask export: {frame_idx} frames in "
                f"{format_duration(video_elapsed)} ({video_rate:.2f} frames/s) -> "
                f"{output_dir}",
                flush=True,
            )
            stage_frame_count += frame_idx
            processed_videos += 1

    if processed_videos == 0:
        raise FileNotFoundError(
            "No mask export inputs were processed. Check that --input-dir contains "
            "camera "
            "empty_frame.png/overlay_mask.png files and that --output-dir contains "
            f"small chunks under camera*/small/ matching {args.chunk_pattern!r}."
        )

    stage_elapsed = time.perf_counter() - stage_started_at
    stage_rate = stage_frame_count / stage_elapsed if stage_elapsed > 0 else 0.0
    print(
        f"Mask export complete: {processed_videos} videos, {stage_frame_count} frames "
        f"in {format_duration(stage_elapsed)} ({stage_rate:.2f} frames/s)",
        flush=True,
    )
