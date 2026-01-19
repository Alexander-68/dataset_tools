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
- `correct_face_keypoints.py`: Merge improved face keypoints into pose labels
  with weighted blending and deviation stats.
- `correct_mpii_keypoints.py`: Replace selected COCO keypoints with MPII pose
  keypoints using bbox-based matching.
- `crop_portrait_square_yolo.py`: Face-centered square crops of portraits using
  YOLO pose keypoints, with optional resize/rotate/debug.
- `crop_to_annotations_yolo.py`: Crop images and YOLO labels around all boxes
  and keypoints with padded aspect ratio selection and prefixed outputs.
- `download_google_images.py`: Download full-resolution images from Google or
  Yandex Images search URLs, save as JPEGs, and optionally resize.
- `extend_flip_yolo.py`: Extend images with a flipped duplicate and update
  bounding boxes/keypoints.
- `extract_tfrecord_images.py`: Extract JPEG images from TFRecord files with
  progress and stats.
- `merge_datasets.py`: Merge multiple datasets into a unified train/val layout
  and update `dataset.yaml` counts.
- `merge_pose_results.py`: Merge body and face pose labels, refining face points.
- `mosaic_self_yolo.py`: Build self-mosaics and rewrite YOLO detection/pose
  labels.
- `mosaic_yolo.py`: Build multi-image mosaics with optional flip/rotate and
  merged labels.
- `resize_images.py`: Resize images (and optionally convert HEIC) with progress
  and stats.
- `rename_images_labels.py`: Rename images with matching labels using a pattern
  and update label filenames.
- `rotate_head_tilt_yolo.py`: Rotate portraits based on head tilt and update
  pose labels.
- `rotate_images_labels.py`: Rotate images to fixed angles and update labels,
  supporting YOLO detection and pose.
- `visualize-pose.py`: Overlay YOLO pose keypoints and boxes onto images for
  quick inspection.

## Docs

Each script has a matching `.md` file in this directory with full usage and
arguments.
