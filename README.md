# MEG Skater Stimulus Pipeline

Prepares synchronized multi-camera videos for MEG stimulus presentation.

## Code

- `stimulus_pipeline.py`: command-line entry point.
- `skater_pipeline/config.py`: configuration parsing and validation.
- `skater_pipeline/video.py`: paths, file discovery, and FFmpeg helpers.
- `skater_pipeline/preprocess.py`: `cut_big` and `cut_small`.
- `skater_pipeline/background.py`: `clean_bg`.
- `skater_pipeline/export_masks.py`: optional `export_masks` diagnostics.
- `skater_pipeline/mask_overlay.py`: combined `mask_overlay` stage.
- `skater_pipeline/static_overlays.py`: transparent PNG overlays.
- `skater_pipeline/camera_cut.py`: camera switching and logs.
- `skater_pipeline/cli.py`: arguments and stage dispatch.
- `crop_keypoints.py`: adjusts HDF5 keypoints for camera crops.

## Setup

```bash
conda env create -f environment.yml
conda activate skater-stimulus-pipeline
```

Copy `pipeline_config.example.json` to `pipeline_config.json` and set the data
location. Check it with:

```bash
python stimulus_pipeline.py --show-paths
```

Path priority: command line, environment, config file, defaults.

## Input

```text
input/
  timing.csv
  color_map.csv                  optional
  video_geometry.json           optional
  static_overlays.json          optional
  static_overlays/<set>/        optional named overlay sets
  cameras/
    camera1/
      raw.mp4
      first_frame.txt
      background_mask.png       optional
      empty_frame.png           optional
      overlay_mask.png          optional
    camera2/
    camera3/
```

Starter files are in `input_template/`.

## Run

```bash
python stimulus_pipeline.py --steps cut_big
python stimulus_pipeline.py --steps cut_small
python stimulus_pipeline.py --steps clean_bg
python stimulus_pipeline.py --steps export_masks
python stimulus_pipeline.py --steps mask_overlay
python stimulus_pipeline.py --steps static_overlays
python stimulus_pipeline.py --steps camera_cut
```

- `preprocess`: runs `cut_big` and `cut_small`.
- `all`: runs the normal workflow without `export_masks`.
- `--dry-run`: prints FFmpeg commands without running them.
- `--chunk-pattern "big1_small*.mp4"`: limits matching stages.
- `--big-index 1 --small-index 2`: selects one chunk.
- `--help`: lists all options.

## Geometry and Keypoints

`input/video_geometry.json` defines per-camera crops and final normalization.
Dimensions must be even; normalization mode is `scale` or `pad`.

```bash
python crop_keypoints.py --geometry input/video_geometry.json /path/to/*.h5
```

The adjusted copies use the suffix `_cropped`.

## Static Overlays

Default config: `input/static_overlays.json`.

Named config: `input/static_overlays/<set>/static_overlays.json`.

```bash
python stimulus_pipeline.py --steps static_overlays --static-overlay-set condition_a
python stimulus_pipeline.py --steps camera_cut --static-overlay-set condition_a
```

See the JSON files under `input_template/static_overlays/` for examples.

## Output

```text
output/
  camera1/{big,small,cleaned,overlaid,static_overlaid}/
  camera2/
  camera3/
  masks/
  final/
```
