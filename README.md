# MEG Skater Stimulus Pipeline

Prepares synchronized multi-camera videos for MEG stimulus presentation.

## Code

- `stimulus_pipeline.py`: command-line entry point.
- `skater_pipeline/config.py`: configuration parsing and validation.
- `skater_pipeline/video.py`: paths, file discovery, and FFmpeg helpers.
- `skater_pipeline/preprocess.py`: `cut_big` and `cut_small`.
- `skater_pipeline/background.py`: `clean_bg`.
- `skater_pipeline/export_masks.py`: optional `export_masks` diagnostics.
- `skater_pipeline/overlays.py`: PNG overlays with optional skater masking.
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
  video_geometry.json           optional
  overlays.json                 optional
  overlays/                     optional named alternatives
    alternative_a/
      overlays.json
      camera1_overlay.png
      camera2_overlay.png
      camera3_overlay.png
    alternative_b/
      overlays.json
      camera1_overlay.png
      camera2_overlay.png
      camera3_overlay.png
  cameras/
    camera1/
      raw.mp4
      first_frame.txt
      background_mask.png       optional
      empty_frame.png           required by masked overlays
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
python stimulus_pipeline.py --steps overlays
python stimulus_pipeline.py --steps camera_cut
```

- `preprocess`: runs `cut_big` and `cut_small`.
- `all`: runs the normal workflow without `export_masks`.
- `--dry-run`: prints FFmpeg commands without running them.
- Long frame-processing stages report progress every 30 seconds.
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

## Overlays

Every overlay image must be a PNG. Default config: `input/overlays.json`.

Each named alternative has its own folder and `overlays.json`. Image paths inside
that JSON are relative to the same folder:

```text
input/overlays/
  alternative_a/
    overlays.json
    camera1_overlay.png
  alternative_b/
    overlays.json
    camera1_overlay.png
```

List the available alternatives, then run one by folder name:

```bash
python stimulus_pipeline.py --list-overlay-sets
python stimulus_pipeline.py --steps overlays camera_cut --overlay-set alternative_a
python stimulus_pipeline.py --steps overlays camera_cut --overlay-set alternative_b
```

Each alternative is written separately—for example,
`output/camera1/overlaid/alternative_a/` and
`output/camera1/overlaid/alternative_b/`—so running one does not overwrite the
other.

Set `"mask_skater": true` on an individual overlay to make that PNG transparent
wherever the skater is detected. Set it to `false` or omit it to draw the PNG
over the skater. Masked and unmasked PNGs can be mixed in the same configuration:

```json
{
  "overlays": [
    {"image": "behind_skater.png", "mask_skater": true},
    {"image": "over_skater.png", "mask_skater": false}
  ]
}
```

Skater detection runs only for chunks with at least one masked overlay. Detection
and PNG compositing happen in one streaming pass, without writing an intermediate
mask video. Use `export_masks` only when diagnostic mask images are needed.

See the `alternative_a` and `alternative_b` JSON files under
`input_template/overlays/` for examples.

## Output

```text
output/
  camera1/{big,small,cleaned,overlaid}/
  camera2/
  camera3/
  masks/
  final/
```
