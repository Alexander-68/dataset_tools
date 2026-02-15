from __future__ import annotations

import argparse
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

try:
    import yt_dlp
except ImportError as exc:  # pragma: no cover - dependency check
    yt_dlp = None
    YT_DLP_IMPORT_ERROR = exc
else:
    YT_DLP_IMPORT_ERROR = None


def format_eta(seconds: float | None) -> str:
    if seconds is None or math.isinf(seconds) or seconds < 0:
        return "ETA --:--"
    total = int(round(seconds))
    if total >= 3600:
        hours = total // 3600
        minutes = (total % 3600) // 60
        secs = total % 60
        return f"ETA {hours:02d}:{minutes:02d}:{secs:02d}"
    minutes = total // 60
    secs = total % 60
    return f"ETA {minutes:02d}:{secs:02d}"


def short_label(text: str, max_len: int = 36) -> str:
    ascii_text = text.encode("ascii", "replace").decode("ascii")
    if len(ascii_text) <= max_len:
        return ascii_text
    return f"...{ascii_text[-(max_len - 3):]}"


def format_speed(speed: float | None) -> str:
    if speed is None or speed <= 0:
        return "--"
    units = ["B/s", "KB/s", "MB/s", "GB/s"]
    value = float(speed)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:0.1f} {unit}"
        value /= 1024
    return "--"


class ProgressTracker:
    def __init__(self, total_urls: int) -> None:
        self.total_urls = total_urls
        self.start_time = time.time()
        self.current_index = 1
        self.current_fraction = 0.0
        self.current_label = ""
        self.current_speed = "--"

    def begin_item(self, index: int, url: str) -> None:
        self.current_index = index
        self.current_fraction = 0.0
        self.current_speed = "--"
        self.current_label = short_label(url)
        self.render()

    def hook(self, status: dict[str, Any]) -> None:
        state = status.get("status")
        if state == "downloading":
            total = status.get("total_bytes") or status.get("total_bytes_estimate")
            downloaded = status.get("downloaded_bytes", 0)
            if total and total > 0:
                self.current_fraction = max(
                    self.current_fraction, min(float(downloaded) / float(total), 1.0)
                )
            filename = status.get("filename")
            if filename:
                self.current_label = short_label(Path(filename).name)
            self.current_speed = format_speed(status.get("speed"))
            self.render()
        elif state == "finished":
            self.current_fraction = 1.0
            filename = status.get("filename")
            if filename:
                self.current_label = short_label(Path(filename).name)
            self.render()

    def mark_done(self) -> None:
        self.current_fraction = 1.0
        self.render()

    def render(self) -> None:
        total = self.total_urls
        if total <= 0:
            return
        completed = max(0, self.current_index - 1)
        overall_fraction = min((completed + self.current_fraction) / total, 1.0)
        bar_width = 30
        filled = int(bar_width * overall_fraction)
        bar = "#" * filled + "-" * (bar_width - filled)
        percent = int(overall_fraction * 100)
        elapsed = time.time() - self.start_time
        eta = None
        if overall_fraction > 0:
            eta = (elapsed / overall_fraction) * (1.0 - overall_fraction)
        sys.stdout.write(
            f"\r[{bar}] {self.current_index}/{total} {percent:3d}% "
            f"{format_eta(eta)} {self.current_speed:>10} {self.current_label}   "
        )
        sys.stdout.flush()


def resolve_cwd_path(path: Path, cwd: Path) -> Path:
    if path.is_absolute():
        return path
    return cwd / path


def load_urls(urls_file: Path) -> list[str]:
    urls: list[str] = []
    with urls_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            urls.append(raw)
    return urls


def sanitize_name_part(value: str, default: str = "video", max_len: int = 80) -> str:
    ascii_value = value.encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", ascii_value).strip("._-")
    if not cleaned:
        cleaned = default
    return cleaned[:max_len]


def pick_first_string(info: dict[str, Any], keys: list[str]) -> str | None:
    for key in keys:
        value = info.get(key)
        if isinstance(value, str):
            trimmed = value.strip()
            if trimmed:
                return trimmed
    return None


def choose_output_basename(info: dict[str, Any], original_url: str) -> str:
    video_id = sanitize_name_part(str(info.get("id") or "video"), default="video")
    extractor = str(info.get("extractor_key") or info.get("extractor") or "").lower()
    webpage_url = str(info.get("webpage_url") or original_url).lower()

    if "youtube" in extractor or "youtu" in webpage_url:
        return video_id

    if "tiktok" in extractor or "tiktok.com" in webpage_url:
        user_value = pick_first_string(
            info,
            ["uploader_id", "uploader", "creator", "channel", "channel_id"],
        )
        user = sanitize_name_part(user_value or "u", default="u", max_len=40)
        return f"{user}_{video_id}"

    return video_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download TikTok or YouTube videos in highest available quality "
            "using yt-dlp."
        )
    )
    parser.add_argument(
        "url",
        nargs="?",
        help="Single video URL to download. If omitted, read from --urls-file.",
    )
    parser.add_argument(
        "--urls-file",
        default="urls.txt",
        help="Text file with one URL per line. Default: urls.txt (in CWD).",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Output directory for downloaded videos (relative to CWD). Default: .",
    )
    parser.add_argument(
        "--cookies",
        default=None,
        help="Optional cookies file path for sites requiring login/session cookies.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=10,
        help="Retry attempts for network/download errors. Default: 10.",
    )
    parser.add_argument(
        "--allow-playlists",
        action="store_true",
        help="Allow playlist URLs. By default, only a single video is downloaded per URL.",
    )
    return parser.parse_args()


def main() -> None:
    if yt_dlp is None:
        raise ImportError(
            "yt_dlp is required. Install it with `pip install yt-dlp`. "
            f"Import error: {YT_DLP_IMPORT_ERROR}"
        )

    args = parse_args()
    if args.retries < 0:
        raise ValueError("--retries must be >= 0")

    cwd = Path.cwd()
    script_dir = Path(__file__).resolve().parent
    urls_file = resolve_cwd_path(Path(args.urls_file), cwd)
    output_dir = resolve_cwd_path(Path(args.output_dir), cwd)
    cookies_path = resolve_cwd_path(Path(args.cookies), cwd) if args.cookies else None

    if args.url:
        urls = [args.url.strip()]
        source = f"single URL from CLI"
    else:
        if not urls_file.exists():
            raise FileNotFoundError(
                f"URLs file not found: {urls_file}. Provide a URL argument or create urls.txt."
            )
        urls = load_urls(urls_file)
        source = f"file: {urls_file}"

    urls = [u for u in urls if u]
    if not urls:
        print("No URLs to download.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    yt_dlp_version = getattr(yt_dlp.version, "__version__", "unknown")
    print(
        "Starting video download with yt-dlp.\n"
        f"- Data root (CWD): {cwd}\n"
        f"- Script dir: {script_dir}\n"
        f"- URL source: {source}\n"
        f"- URLs to process: {len(urls)}\n"
        f"- Output dir: {output_dir}\n"
        f"- Cookies file: {cookies_path if cookies_path else 'none'}\n"
        f"- Retries: {args.retries}\n"
        f"- Allow playlists: {args.allow_playlists}\n"
        f"- yt-dlp version: {yt_dlp_version}\n"
        "- Format target: bestvideo+bestaudio (fallback: best), merge to mp4\n"
        "- Filename policy: YouTube -> <video_id>; TikTok -> <user>_<video_id>"
    )

    tracker = ProgressTracker(total_urls=len(urls))
    total = len(urls)
    success = 0
    failed = 0
    start_time = time.time()

    for index, url in enumerate(urls, start=1):
        tracker.begin_item(index, url)
        common_opts: dict[str, Any] = {
            "format": "bestvideo*+bestaudio/best",
            "merge_output_format": "mp4",
            "noplaylist": not args.allow_playlists,
            "retries": args.retries,
            "fragment_retries": args.retries,
            "windowsfilenames": True,
            "quiet": True,
            "no_warnings": True,
        }
        if cookies_path:
            common_opts["cookiefile"] = str(cookies_path)
        try:
            with yt_dlp.YoutubeDL(common_opts) as probe:
                info = probe.extract_info(url, download=False)
            if info is None:
                raise RuntimeError("Could not read video metadata.")
            basename = choose_output_basename(info, url)

            ydl_opts = dict(common_opts)
            ydl_opts["outtmpl"] = str(output_dir / f"{basename}.%(ext)s")
            ydl_opts["progress_hooks"] = [tracker.hook]
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            success += 1
        except Exception as exc:  # pragma: no cover - network dependent
            failed += 1
            print(f"\nFailed [{index}/{total}]: {url}\n- Error: {exc}")
        finally:
            tracker.mark_done()

    elapsed = time.time() - start_time
    print(
        "\nDone.\n"
        f"- URLs processed: {total}\n"
        f"- Successful: {success}\n"
        f"- Failed: {failed}\n"
        f"- Elapsed: {elapsed:0.1f}s\n"
        f"- Output dir: {output_dir}"
    )


if __name__ == "__main__":
    main()
