# download_videos_yt_dlp.py

Download TikTok or YouTube videos in the highest available quality using
`yt_dlp`. By default, the script reads URLs from `urls.txt` in the current
working directory (one URL per line) and saves videos to the current folder.

It can also download a single URL passed on the command line.

## Usage

```bash
python download_videos_yt_dlp.py
```

```bash
python download_videos_yt_dlp.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
```

```bash
python download_videos_yt_dlp.py --urls-file my_urls.txt --output-dir downloads
```

## Arguments

- `url`: Optional single video URL. If provided, `--urls-file` is ignored.
- `--urls-file`: Text file with one URL per line. Default: `urls.txt` (in CWD).
- `--output-dir`: Output folder for downloaded files. Default: current folder (`.`).
- `--cookies`: Optional cookies file path for sites that require login/session cookies.
- `--retries`: Retry attempts for network/download errors. Default: `10`.
- `--allow-playlists`: Allow playlist URLs. Default behavior downloads one video per URL.

## Output

At startup, the script prints what it will do and all parameter values. During
download, it shows a progress bar with ETA. At the end, it reports processed,
successful, and failed URL counts.

## Paths

- Data root: current working directory (CWD). Relative paths resolve from CWD.
- Script directory: contains this script and its `.md` description.

## Requirements

- `yt_dlp` installed (example: `pip install yt-dlp`)

## Notes

- The script requests best available format with `bestvideo+bestaudio`, then
  falls back to `best` if needed.
- Filename policy:
  - YouTube: `<video_id>.<ext>`
  - TikTok: `<user>_<video_id>.<ext>` (fallback user is `u`)
