"""Stages that cut raw recordings into synchronized, manageable clips."""

import argparse
import math

from .config import load_video_geometry
from .video import (
    best_small_chunk_duration,
    big_chunk_dir,
    crop_filter,
    get_video_duration,
    require_tool,
    run_command,
    small_chunk_dir,
    synced_chunk_plan,
)

def cut_big_chunks(args: argparse.Namespace) -> None:
    """Cut long raw camera recordings into synchronized big chunks."""
    require_tool("ffmpeg")
    geometry = load_video_geometry(args.input_dir)
    for camera_plan in synced_chunk_plan(args.input_dir, args.fps, args.lead_in_seconds):
        for item in camera_plan:
            if args.big_index is not None and item["chunk"] != args.big_index:
                continue
            camera_name = str(item["camera_name"])
            camera_dir = big_chunk_dir(args, str(item["camera_name"]))
            output_file = camera_dir / f"big{item['chunk']}.mp4"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            crop = geometry.crops.get(camera_name)
            cmd = [
                "ffmpeg",
                "-y",
                "-ss",
                str(item["start"]),
                "-to",
                str(item["end"]),
                "-i",
                str(item["video_file"]),
            ]
            if crop is not None:
                cmd.extend(["-vf", crop_filter(crop)])
            cmd.extend(
                [
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "18",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    str(output_file),
                ]
            )
            run_command(cmd, dry_run=args.dry_run)


def cut_small_chunks(args: argparse.Namespace) -> None:
    """Split big chunks into smaller clips for inspection and later editing."""
    require_tool("ffmpeg")
    big_chunks = sorted(args.output_dir.glob("*/big/big*.mp4"))
    if args.big_index is not None:
        big_chunks = [
            path
            for path in big_chunks
            if path.stem == f"big{args.big_index}"
        ]
    if not big_chunks:
        raise FileNotFoundError(f"No big chunks found under: {args.output_dir}")

    if args.small_duration is None:
        durations = [get_video_duration(path) for path in big_chunks]
        small_duration = best_small_chunk_duration(
            durations, args.small_duration_min, args.small_duration_max
        )
    else:
        small_duration = args.small_duration
    print(f"Small chunk duration: {small_duration:.2f}s")

    for big_chunk_file in big_chunks:
        camera_name = big_chunk_file.parent.parent.name
        output_dir = small_chunk_dir(args, camera_name)
        output_dir.mkdir(parents=True, exist_ok=True)
        duration = get_video_duration(big_chunk_file)
        num_chunks = int(math.floor(duration / small_duration))
        for index in range(num_chunks):
            if args.small_index is not None and index + 1 != args.small_index:
                continue
            start = index * small_duration
            end = start + small_duration
            output_file = output_dir / f"{big_chunk_file.stem}_small{index + 1}.mp4"
            cmd = [
                "ffmpeg",
                "-y",
                "-ss",
                str(start),
                "-to",
                str(end),
                "-i",
                str(big_chunk_file),
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                str(output_file),
            ]
            run_command(cmd, dry_run=args.dry_run)
