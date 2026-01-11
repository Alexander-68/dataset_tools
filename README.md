# YOLO Dataset Utilities

Small, focused scripts for preparing YOLO detection/pose datasets: annotation
generation, label cleanup/merging, mosaics, and image preprocessing.

## Conventions

- Current Working Directory (CWD): where your `images/` and `labels/` live.
- Script Directory: where the scripts, their `.md` docs, and `.pt` models live.
- Most tools read/write relative to CWD unless a path is provided.

## Tools

- `annotate_images.py`: Run a YOLO pose model on images and write pose labels,
  merging normal and flipped predictions.
- `cleanup_labels.py`: Move label files without matching images into `labels-x`.
- `correct_keypoints.py`: Normalize person keypoint visibility flags and zero
  coordinates when invisible.
- `crop_portrait_square_yolo.py`: Face-centered square crops of portraits using
  YOLO pose keypoints, with optional resize/rotate/debug.
- `extend_flip_yolo.py`: Extend images with a flipped duplicate and update
  bounding boxes/keypoints.
- `merge_datasets.py`: Merge multiple datasets into a unified train/val layout
  and update `dataset.yaml` counts.
- `merge_pose_results.py`: Merge body and face pose labels, refining face points.
- `mosaic_self_yolo.py`: Build self-mosaics and rewrite YOLO detection/pose
  labels.
- `mosaic_yolo.py`: Build multi-image mosaics with optional flip/rotate and
  merged labels.
- `resize_images.py`: Resize images (and optionally convert HEIC) with progress
  and stats.
- `rotate_head_tilt_yolo.py`: Rotate portraits based on head tilt and update
  pose labels.

## Docs

Each script has a matching `.md` file in this directory with full usage and
arguments.
