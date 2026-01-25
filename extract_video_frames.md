# extract_video_frames.py

Extract frames from all videos in the current working directory. Each video
creates a folder with the same name as the video file stem, with frames saved
under an `images/` subfolder as JPEG files named `video-000123.jpg` using the
original frame number and zero-padding based on the total frame count.

## Usage

```bash
python extract_video_frames.py
```

```bash
python extract_video_frames.py --skip 5
```

```bash
python extract_video_frames.py --crop-top-bottom 48
```

## Arguments

- `--skip`: Number of frames to skip between saves. Default: 0 (save every
  frame). For example, `--skip 5` saves 1 frame then skips 5 frames.
- `--crop-top-bottom`: Pixels to crop from both the top and bottom of each
  frame. Default: 96.

## Notes

- Requires OpenCV: `pip install opencv-python`.
- Processes common video extensions: `.mp4`, `.mov`, `.mkv`, `.avi`, `.webm`,
  `.mpg`, `.mpeg`, `.m4v`, `.wmv`.
