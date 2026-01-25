from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

try:
    import cv2
except ImportError as exc:  # pragma: no cover - optional dependency
    cv2 = None
    CV2_IMPORT_ERROR = exc
else:
    CV2_IMPORT_ERROR = None


VIDEO_EXTENSIONS = {
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".webm",
    ".wmv",
}


def format_eta(seconds: float | None) -> str:
    if seconds is None or math.isinf(seconds) or seconds < 0:
        return "ETA --:--"
    total_seconds = int(round(seconds))
    if total_seconds >= 3600:
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        return f"ETA {hours:02d}:{minutes:02d}:{secs:02d}"
    minutes = total_seconds // 60
    secs = total_seconds % 60
    return f"ETA {minutes:02d}:{secs:02d}"


def render_progress(
    current: int, total: int | None, start_time: float, label: str
) -> None:
    bar_width = 30
    if total and total > 0:
        filled = int(bar_width * current / total)
        bar = "#" * filled + "-" * (bar_width - filled)
        percent = int(100 * current / total)
        elapsed = time.time() - start_time
        eta = None
        if current > 0:
            eta = (elapsed / current) * (total - current)
        tail = label
        if len(tail) > 34:
            tail = f"...{tail[-31:]}"
        sys.stdout.write(
            f"\r[{bar}] {current}/{total} {percent:3d}% {format_eta(eta)} {tail}   "
        )
    else:
        bar = "#" * min(current, bar_width)
        elapsed = time.time() - start_time
        sys.stdout.write(
            f"\r[{bar:<30}] {current} frames "
            f"{format_eta(None)} elapsed {elapsed:0.1f}s {label}   "
        )
    sys.stdout.flush()


def iter_videos(root: Path) -> list[Path]:
    return [
        path
        for path in sorted(root.iterdir())
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]


def extract_frames(root: Path, skip: int, crop_top_bottom: int) -> None:
    if cv2 is None:
        raise ImportError(
            "OpenCV is required. Install it with `pip install opencv-python`. "
            f"Import error: {CV2_IMPORT_ERROR}"
        )

    videos = iter_videos(root)
    print(
        "Starting video frame extraction.\n"
        f"- Data root (CWD): {root}\n"
        f"- Videos found: {len(videos)}\n"
        f"- Skip: {skip} (save 1, skip {skip})\n"
        f"- Crop top/bottom: {crop_top_bottom} px\n"
        f"- Extensions: {', '.join(sorted(VIDEO_EXTENSIONS))}"
    )

    if not videos:
        print("No videos found to process.")
        return

    total_videos = len(videos)
    total_frames = 0
    total_saved = 0
    total_failed = 0
    total_crop_skipped = 0

    for video_index, video_path in enumerate(videos, start=1):
        output_dir = root / video_path.stem / "images"
        output_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Could not open video: {video_path}")
            total_failed += 1
            continue

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_index = 0
        saved_for_video = 0
        start_time = time.time()
        pad_width = len(str(frame_count)) if frame_count > 0 else 0
        print(
            f"\n[{video_index}/{total_videos}] {video_path.name} "
            f"(frames: {frame_count if frame_count > 0 else 'unknown'})"
        )

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_number = frame_index + 1
            save_frame = (frame_index % (skip + 1)) == 0
            if save_frame:
                if pad_width:
                    frame_label = f"{frame_number:0{pad_width}d}"
                else:
                    frame_label = str(frame_number)
                output_name = f"{video_path.stem}-{frame_label}.jpg"
                output_path = output_dir / output_name
                working_frame = frame
                if crop_top_bottom > 0:
                    height = frame.shape[0]
                    if height <= crop_top_bottom * 2:
                        total_crop_skipped += 1
                        frame_index += 1
                        render_progress(
                            frame_index,
                            frame_count if frame_count > 0 else None,
                            start_time,
                            video_path.name,
                        )
                        continue
                    working_frame = frame[crop_top_bottom : height - crop_top_bottom, :, :]
                success = cv2.imwrite(
                    str(output_path),
                    working_frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 95],
                )
                if success:
                    total_saved += 1
                    saved_for_video += 1
                else:
                    total_failed += 1
            frame_index += 1
            render_progress(frame_index, frame_count if frame_count > 0 else None, start_time, video_path.name)

        cap.release()
        total_frames += frame_index
        print(f"\nSaved {saved_for_video} frames to {output_dir}")

    print(
        "\nDone.\n"
        f"- Videos processed: {total_videos}\n"
        f"- Total frames read: {total_frames}\n"
        f"- Frames saved: {total_saved}\n"
        f"- Failed writes: {total_failed}\n"
        f"- Crop skipped (too short): {total_crop_skipped}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract frames from all videos in the current folder."
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Number of frames to skip between saves. Default: 0 (save every frame).",
    )
    parser.add_argument(
        "--crop-top-bottom",
        type=int,
        default=96,
        help="Crop pixels from both top and bottom of each frame. Default: 96.",
    )

    args = parser.parse_args()
    if args.skip < 0:
        raise ValueError("--skip must be >= 0")
    if args.crop_top_bottom < 0:
        raise ValueError("--crop-top-bottom must be >= 0")

    cwd = Path.cwd()
    script_dir = Path(__file__).resolve().parent
    print(f"Data root (CWD): {cwd}")
    print(f"Script dir: {script_dir}")
    extract_frames(cwd, args.skip, args.crop_top_bottom)


if __name__ == "__main__":
    main()
