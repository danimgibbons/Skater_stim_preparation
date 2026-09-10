"""Final stage that joins clips while switching between camera viewpoints."""

import argparse
import csv
import fnmatch
import random
from pathlib import Path

from .config import Normalization, VIDEO_EXTENSIONS, load_video_geometry
from .video import (
    base_chunk_name,
    camera_dirs,
    chunk_sort_key,
    get_video_duration,
    normalization_filter,
    require_tool,
    run_command,
    small_chunk_dir,
    static_overlay_output_dir,
    static_overlay_output_path,
    static_overlay_output_suffix,
)

def camera_switch_candidates(
    args: argparse.Namespace,
    camera_name: str,
    base: str,
) -> list[Path]:
    """Return source candidates for one switched-viewpoint clip."""
    static_overlay_path = static_overlay_output_path(
        args,
        camera_name,
        base,
        args.static_overlay_set,
    )
    if args.static_overlay_set is not None:
        return [static_overlay_path]
    non_static_candidates = [
        args.output_dir / camera_name / "overlaid" / f"{base}_cleaned_overlay.mp4",
        args.output_dir / camera_name / "overlaid" / f"{base}_overlay.mp4",
        args.output_dir / camera_name / "cleaned" / f"{base}_cleaned.mp4",
        small_chunk_dir(args, camera_name) / f"{base}.mp4",
    ]
    if args.no_static_overlays:
        return non_static_candidates
    return [
        static_overlay_path,
        *non_static_candidates,
    ]


def camera_switch_source(args: argparse.Namespace, camera_name: str, base: str) -> Path:
    """Select the best available source clip for final viewpoint switching."""
    for path in camera_switch_candidates(args, camera_name, base):
        if path.exists():
            return path
    raise FileNotFoundError(f"No switch source found for {camera_name}, {base}")


def requested_camera_switch_bases(args: argparse.Namespace) -> list[str] | None:
    """Return exact camera-cut chunks requested on the command line, if any."""
    if args.camera_cut_chunks:
        requested = []
        seen = set()
        for value in args.camera_cut_chunks:
            base = base_chunk_name(Path(value).stem)
            if base not in seen:
                requested.append(base)
                seen.add(base)
        return requested

    if args.big_index is not None and args.small_index is not None:
        return [f"big{args.big_index}_small{args.small_index}"]

    return None


def camera_switch_bases_for_camera(args: argparse.Namespace, camera_name: str) -> set[str]:
    """Return switchable small-chunk stems available for one camera."""
    bases = set()
    if args.static_overlay_set is not None:
        source_dirs = [
            static_overlay_output_dir(
                args,
                camera_name,
                args.static_overlay_set,
            )
        ]
    else:
        source_dirs = [
            args.output_dir / camera_name / "overlaid",
            args.output_dir / camera_name / "cleaned",
            small_chunk_dir(args, camera_name),
        ]
        if not args.no_static_overlays:
            source_dirs.insert(0, static_overlay_output_dir(args, camera_name, None))
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


def camera_switch_bases(args: argparse.Namespace, camera_names: list[str]) -> list[str]:
    """Return the small-chunk stems to process during final camera switching."""
    requested = requested_camera_switch_bases(args)
    if requested is not None:
        return requested

    bases_by_camera = {
        camera_name: camera_switch_bases_for_camera(args, camera_name)
        for camera_name in camera_names
    }
    common_bases = set.intersection(*bases_by_camera.values()) if bases_by_camera else set()
    if not common_bases:
        counts = ", ".join(
            f"{camera_name}: {len(bases)}"
            for camera_name, bases in bases_by_camera.items()
        )
        if args.static_overlay_set is not None:
            raise FileNotFoundError(
                "No complete camera-cut chunks found for static overlay set "
                f"{args.static_overlay_set!r} matching {args.chunk_pattern!r}. "
                "Run the static overlay stage with the same --static-overlay-set "
                f"first. Available per camera: {counts or '(none)'}."
            )
        raise FileNotFoundError(
            "No complete camera-cut chunks found across all cameras matching "
            f"{args.chunk_pattern!r}. Available per camera: {counts or '(none)'}."
        )

    incomplete_bases = set.union(*bases_by_camera.values()) - common_bases
    if incomplete_bases:
        skipped = sorted(incomplete_bases, key=chunk_sort_key)
        preview = ", ".join(skipped[:10])
        suffix = "" if len(skipped) <= 10 else f", and {len(skipped) - 10} more"
        print(f"Skipping incomplete camera_cut chunks: {preview}{suffix}")

    return sorted(common_bases, key=chunk_sort_key)


def create_camera_switch_video_for_base(
    args: argparse.Namespace,
    base: str,
    camera_names: list[str],
    rng: random.Random,
    normalization: Normalization | None,
) -> None:
    """Create one switched-viewpoint stimulus video and frame-level camera log."""
    files = [
        camera_switch_source(args, camera_name, base)
        for camera_name in camera_names
    ]
    output_dir = args.output_dir / "final"
    output_stem = f"{base}_switch_{args.cut_frames}f"
    if args.static_overlay_set is not None:
        suffix = static_overlay_output_suffix(args.static_overlay_set).lstrip("_")
        output_stem = f"{output_stem}_{suffix}"
    if args.camera_cut_mode != "random":
        output_stem = f"{output_stem}_{args.camera_cut_mode}"
    if args.final_label is not None:
        output_stem = f"{output_stem}_{args.final_label}"
    output_video = output_dir / f"{output_stem}.mp4"
    output_log = output_video.with_name(f"{output_video.stem}_camlog.csv")

    durations = [get_video_duration(file_path) for file_path in files]
    duration = min(durations)
    total_frames = int(duration * args.fps)
    cameras = list(range(len(files)))
    if len(cameras) < 2:
        raise ValueError("Camera switching requires at least two camera source clips.")
    if total_frames <= 0:
        raise ValueError(f"Switch source duration is empty for: {base}")

    segments = []

    frame = 0
    if args.camera_cut_mode == "cycle":
        camera_index = 0
        while frame < total_frames:
            end = min(frame + args.cut_frames, total_frames)
            segments.append((frame, end, cameras[camera_index]))
            camera_index = (camera_index + 1) % len(cameras)
            frame = end
    else:
        current_camera = rng.choice(cameras)
        while frame < total_frames:
            end = min(frame + args.cut_frames, total_frames)
            segments.append((frame, end, current_camera))
            current_camera = rng.choice([c for c in cameras if c != current_camera])
            frame = end

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        with output_log.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "camera"])
            for start, end, camera in segments:
                for frame_idx in range(start, end):
                    writer.writerow([frame_idx + 1, camera + 1])

    filters = []
    concat_inputs = []
    for i, (start, end, camera) in enumerate(segments):
        t_start = start / args.fps
        t_end = end / args.fps
        label = f"v{i}"
        segment_filters = [
            f"[{camera}:v]trim=start={t_start}:end={t_end}",
            "setpts=PTS-STARTPTS",
        ]
        if normalization is not None:
            segment_filters.append(normalization_filter(normalization))
        filters.append(",".join(segment_filters) + f"[{label}]")
        concat_inputs.append(f"[{label}]")

    filter_complex = (
        ";\n".join(filters)
        + ";\n"
        + "".join(concat_inputs)
        + f"concat=n={len(segments)}:v=1:a=0[outv]"
    )

    cmd = ["ffmpeg", "-y"]
    for file_path in files:
        cmd.extend(["-i", str(file_path)])
    cmd.extend(
        [
            "-filter_complex",
            filter_complex,
            "-map",
            "[outv]",
            "-r",
            str(args.fps),
            "-pix_fmt",
            "yuv420p",
            str(output_video),
        ]
    )
    run_command(cmd, dry_run=args.dry_run)
    if args.dry_run:
        print(f"Would save camera log to: {output_log}")
    else:
        print(f"Saved camera log to: {output_log}")


def create_camera_switch_video(args: argparse.Namespace) -> None:
    """Create final stimulus videos and frame-level camera logs."""
    require_tool("ffmpeg")
    geometry = load_video_geometry(args.input_dir)
    camera_names = [camera_dir.name for camera_dir in camera_dirs(args.input_dir)]
    bases = camera_switch_bases(args, camera_names)
    rng = random.Random(args.seed)

    for base in bases:
        create_camera_switch_video_for_base(
            args,
            base,
            camera_names,
            rng,
            geometry.normalization,
        )
