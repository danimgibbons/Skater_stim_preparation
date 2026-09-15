"""Replace masked background pixels with a clean reference frame."""

import argparse
import subprocess
import time

from .config import (
    PROGRESS_INTERVAL_SECONDS,
    format_duration,
    load_video_geometry,
    progress_summary,
)
from .video import (
    camera_dirs,
    camera_output_root,
    crop_image_array,
    get_video_info,
    require_tool,
    small_chunk_dir,
)


def clean_background(args: argparse.Namespace) -> None:
    """Replace masked background pixels with a clean reference frame."""
    import numpy as np
    from PIL import Image

    require_tool("ffmpeg")
    require_tool("ffprobe")
    geometry = load_video_geometry(args.input_dir)

    for camera_input_dir in camera_dirs(args.input_dir):
        clean_frame_path = camera_input_dir / "empty_frame.png"
        background_mask_path = camera_input_dir / "background_mask.png"
        if not clean_frame_path.exists() or not background_mask_path.exists():
            print(
                f"Skipping {camera_input_dir.name}: "
                "no empty_frame.png/background_mask.png"
            )
            continue

        camera_output_dir = camera_output_root(args, camera_input_dir.name)
        small_dir = small_chunk_dir(args, camera_input_dir.name)
        cleaned_dir = camera_output_dir / "cleaned"
        cleaned_dir.mkdir(parents=True, exist_ok=True)
        clean = np.asarray(Image.open(clean_frame_path).convert("RGB"), dtype=np.float32)
        mask = np.asarray(Image.open(background_mask_path).convert("L"), dtype=np.float32) / 255.0
        crop = geometry.crops.get(camera_input_dir.name)
        clean = crop_image_array(clean, crop)
        mask = crop_image_array(mask, crop)
        mask = mask[..., None]

        for video_path in sorted(small_dir.glob(args.chunk_pattern)):
            output_path = cleaned_dir / f"{video_path.stem}_cleaned.mp4"
            width, height, fps = get_video_info(video_path)
            if clean.shape[:2] != (height, width):
                raise ValueError(f"Clean image size does not match video: {video_path}")
            if mask.shape[:2] != (height, width):
                raise ValueError(f"Background mask size does not match video: {video_path}")

            print(f"Cleaning background: {video_path} -> {output_path}")
            in_pipe = subprocess.Popen(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-nostats",
                    "-i",
                    str(video_path),
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-",
                ],
                stdout=subprocess.PIPE,
            )
            out_pipe = subprocess.Popen(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-nostats",
                    "-y",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-s",
                    f"{width}x{height}",
                    "-r",
                    str(fps),
                    "-i",
                    "-",
                    "-c:v",
                    "libx264",
                    "-crf",
                    str(args.clean_crf),
                    "-preset",
                    args.clean_preset,
                    "-pix_fmt",
                    "yuv420p",
                    str(output_path),
                ],
                stdin=subprocess.PIPE,
            )

            frame_size = width * height * 3
            frame_count = 0
            video_started_at = time.perf_counter()
            last_progress_at = video_started_at
            if in_pipe.stdout is None or out_pipe.stdin is None:
                raise RuntimeError("Could not open ffmpeg pipes.")

            while True:
                raw = in_pipe.stdout.read(frame_size)
                if len(raw) != frame_size:
                    break
                frame = np.frombuffer(raw, np.uint8).reshape((height, width, 3))
                frame = frame.astype(np.float32)
                result = frame * mask + clean * (1 - mask)
                out_pipe.stdin.write(np.clip(result, 0, 255).astype(np.uint8).tobytes())
                frame_count += 1
                now = time.perf_counter()
                if now - last_progress_at >= PROGRESS_INTERVAL_SECONDS:
                    print(
                        progress_summary(
                            "  Cleaning progress",
                            frame_count,
                            0,
                            video_started_at,
                        ),
                        flush=True,
                    )
                    last_progress_at = now

            in_pipe.stdout.close()
            out_pipe.stdin.close()
            input_returncode = in_pipe.wait()
            output_returncode = out_pipe.wait()
            if input_returncode != 0 or output_returncode != 0:
                raise RuntimeError(f"FFmpeg failed while cleaning: {video_path}")
            elapsed = time.perf_counter() - video_started_at
            rate = frame_count / elapsed if elapsed > 0 else 0.0
            print(
                f"Finished cleaning: {frame_count} frames in "
                f"{format_duration(elapsed)} ({rate:.2f} frames/s) -> {output_path}"
            )
