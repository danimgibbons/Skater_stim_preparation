#!/usr/bin/env python3
"""Create cropped copies of HDF5 keypoint files.

The current keypoint files store `/kpts` as `(frames, 34)`, with 17 x columns
followed by 17 y columns. Cropping a video changes the coordinate origin, so the
matching keypoints need the crop x/y offsets subtracted.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy HDF5 keypoint files and shift /kpts into cropped-frame coordinates."
    )
    parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="Input .h5 keypoint files. Camera id is inferred from cam1/cam2/cam3 in the filename.",
    )
    parser.add_argument(
        "--geometry",
        type=Path,
        default=Path("input_template/video_geometry.json"),
        help="JSON file containing cameras.cameraN.crop.x/y values.",
    )
    parser.add_argument(
        "--suffix",
        default="_cropped",
        help="Suffix inserted before .h5 for output files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output file.",
    )
    return parser.parse_args()


def load_crops(geometry_path: Path) -> dict[str, tuple[float, float]]:
    with geometry_path.open("r", encoding="utf-8") as handle:
        geometry = json.load(handle)

    crops: dict[str, tuple[float, float]] = {}
    for camera_id, config in geometry["cameras"].items():
        crop = config["crop"]
        crops[camera_id] = (float(crop["x"]), float(crop["y"]))
    return crops


def infer_camera_id(path: Path) -> str:
    match = re.search(r"cam(?:era)?(\d+)", path.stem, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not infer camera id from filename: {path.name}")
    return f"camera{int(match.group(1))}"


def output_path_for(source: Path, suffix: str) -> Path:
    return source.with_name(f"{source.stem}{suffix}{source.suffix}")


def shifted_keypoints(data: np.ndarray, x_offset: float, y_offset: float) -> np.ndarray:
    shifted = np.array(data, copy=True)

    if shifted.ndim == 2 and shifted.shape[1] == 34:
        shifted[:, :17] -= x_offset
        shifted[:, 17:34] -= y_offset
        return shifted

    if shifted.ndim == 3 and shifted.shape[-1] == 2:
        shifted[..., 0] -= x_offset
        shifted[..., 1] -= y_offset
        return shifted

    if shifted.ndim == 3 and shifted.shape[0] == 17 and shifted.shape[1] == 2:
        shifted[:, 0, :] -= x_offset
        shifted[:, 1, :] -= y_offset
        return shifted

    raise ValueError(f"Unsupported /kpts shape: {data.shape}")


def crop_file(source: Path, crops: dict[str, tuple[float, float]], suffix: str, overwrite: bool) -> Path:
    source = source.expanduser().resolve()
    camera_id = infer_camera_id(source)
    if camera_id not in crops:
        raise KeyError(f"No crop values found for {camera_id}")

    destination = output_path_for(source, suffix)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {destination}")

    x_offset, y_offset = crops[camera_id]
    shutil.copy2(source, destination)

    with h5py.File(destination, "r+") as handle:
        if "kpts" not in handle:
            raise KeyError(f"{source} does not contain a /kpts dataset")

        dataset = handle["kpts"]
        before = dataset[...]
        after = shifted_keypoints(before, x_offset, y_offset)
        dataset[...] = after.astype(dataset.dtype, copy=False)

        handle.attrs["crop_source_file"] = str(source)
        handle.attrs["crop_camera_id"] = camera_id
        handle.attrs["crop_x_offset"] = x_offset
        handle.attrs["crop_y_offset"] = y_offset

    return destination


def main() -> None:
    args = parse_args()
    crops = load_crops(args.geometry)

    for source in args.files:
        destination = crop_file(source, crops, args.suffix, args.overwrite)
        camera_id = infer_camera_id(source)
        x_offset, y_offset = crops[camera_id]
        print(f"{source} -> {destination} ({camera_id}: x -= {x_offset:g}, y -= {y_offset:g})")


if __name__ == "__main__":
    main()
