# Download Google Images Script

`download_google_images.py` downloads full-resolution images from a Google Images
search URL, saves them as JPEGs, and optionally resizes them so the longest
side is at most a target size.

## Features

-   **Full-resolution URLs:** Extracts original image URLs from the search page
    (avoids thumbnails). If direct URLs are missing, it follows result pages and
    uses `og:image`/`twitter:image` metadata.
-   **Resize on save:** Scales down images larger than a maximum dimension while
    preserving aspect ratio.
-   **JPEG output:** Saves all images as `.jpg` with a consistent prefix.
-   **Progress + ETA:** Shows a progress bar with an estimated time remaining.
-   **Stats:** Reports attempted, failed, skipped, and resized counts.
-   **Resume-friendly:** Tracks downloaded URLs so future runs skip duplicates.
-   **Pagination:** Automatically paginates results to reach the target count.

## Usage

```bash
python download_google_images.py --url "<GOOGLE_IMAGES_SEARCH_URL>"
```

### Arguments

-   `--url`: Google Images search URL to scrape (required).
-   `--count`: Number of images to download. Default: `100`.
-   `--output`: Output directory (relative to CWD). Default: `./images`.
-   `--max-dim`: Resize images so the longest side is <= this value. Default: `1024`.
    Set to `0` to disable resizing.
-   `--prefix`: Filename prefix for saved images. Default: `google`.
-   `--timeout`: Request timeout in seconds. Default: `15`.
-   `--delay`: Delay between downloads in seconds. Default: `0.1`.
-   `--user-agent`: User-Agent header for requests.
-   `--max-pages`: Maximum number of search result pages to fetch. `0` means unlimited.
-   `--page-size`: Results per page for pagination (requested via `num=`). Default: `20`.
-   `--state-file`: Path to a text file for tracking downloaded image URLs.
    Defaults to `<output>/<prefix>_downloaded.txt`.
-   `--no-state`: Disable URL tracking state file.

### Output
At startup, the script prints what it will do and the parameter values. During
processing, it shows a progress bar with ETA. When finished, it reports counts
for attempted, failed, skipped, and resized images.

### Examples

**1. Download 100 images to `images/` with default resize to 1024px:**

```bash
python download_google_images.py --url "https://www.google.com/search?tbm=isch&q=blue+jay"
```

**2. Download 30 images and keep original resolution:**

```bash
python download_google_images.py --url "https://www.google.com/search?tbm=isch&q=golden+retriever" --count 30 --max-dim 0
```

**3. Save into a custom folder with a custom prefix:**

```bash
python download_google_images.py --url "https://www.google.com/search?tbm=isch&q=street+art" --output downloads --prefix streetart
```

**4. Resume later with URL tracking (default behavior):**

```bash
python download_google_images.py --url "https://www.google.com/search?tbm=isch&q=pills" --count 100
```

**5. Use pagination limits (optional):**

```bash
python download_google_images.py --url "https://www.google.com/search?tbm=isch&q=pills" --count 100 --max-pages 10
```

Note: Google often caps image results per page to ~20 regardless of `num`, so
larger values may not increase per-page yield.

**6. Use a custom state file or disable tracking:**

```bash
python download_google_images.py --url "https://www.google.com/search?tbm=isch&q=pills" --state-file downloads\\pills_urls.txt
python download_google_images.py --url "https://www.google.com/search?tbm=isch&q=pills" --no-state
```

## Paths

-   Data root: current working directory (CWD). Relative input/output paths resolve from CWD.
-   Script directory: contains the script and its `.md` description.

## Dependencies

-   `requests`
-   `Pillow` (PIL)

To install dependencies:

```bash
pip install requests Pillow
```

## Notes

-   Google Images results can change and may block requests. If you encounter
    failures, try increasing `--delay` or customizing `--user-agent`.
-   If you see "Found 0 candidate image URLs", Google may have returned a
    consent page or a restricted response. Opening the URL once in a browser
    or adding `&hl=en` can help.
-   If downloads stall with `non_image` counts, Google likely served HTML pages
    instead of image bytes; try adding `&hl=en` or `&gl=us`.
-   The script retries with a lightweight user-agent to fetch basic HTML results
    when Google returns a JS-only page.
