from __future__ import annotations

import argparse
import html
import re
import sys
import time
import warnings
from io import BytesIO
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlencode, urljoin, urlparse

import requests
from PIL import Image, ImageOps, UnidentifiedImageError


def resolve_path(path: Path, base: Path) -> Path:
    if path.is_absolute():
        return path
    return base / path


def format_eta(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "ETA --:--"
    total = int(round(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    if hours:
        return f"ETA {hours:d}:{minutes:02d}:{secs:02d}"
    return f"ETA {minutes:02d}:{secs:02d}"


def render_progress(current: int, total: int, eta: str, name: str) -> None:
    if total <= 0:
        return
    bar_width = 30
    filled = int(bar_width * current / total)
    bar = "#" * filled + "-" * (bar_width - filled)
    percent = int(100 * current / total)
    tail = name.encode("ascii", "replace").decode("ascii")
    if len(tail) > 40:
        tail = f"...{tail[-37:]}"
    sys.stdout.write(
        f"\r[{bar}] {current}/{total} {percent:3d}% {eta} {tail}   "
    )
    sys.stdout.flush()


def clean_url(raw: str) -> str:
    url = raw.replace("\\u003d", "=").replace("\\u0026", "&")
    url = url.replace("\\/", "/")
    url = unquote(url)
    if url.startswith("//"):
        url = "https:" + url
    return url


def normalize_image_url(url: str) -> str:
    parsed = urlparse(url)
    if "google." in parsed.netloc and parsed.path.startswith("/imgres"):
        params = parse_qs(parsed.query)
        img_url = params.get("imgurl")
        if img_url:
            return clean_url(img_url[0])
    return url


def add_query_param(url: str, key: str, value: str) -> str:
    parsed = urlparse(url)
    params = parse_qs(parsed.query)
    params[key] = [value]
    new_query = urlencode(params, doseq=True)
    return parsed._replace(query=new_query).geturl()


def is_thumbnail_url(url: str) -> bool:
    lower = url.lower()
    if lower.startswith("data:"):
        return True
    if "encrypted-tbn0" in lower:
        return True
    if "gstatic.com/images" in lower and "tbn" in lower:
        return True
    if "googleusercontent.com" in lower and "tbn" in lower:
        return True
    return False


def extract_image_urls(html: str) -> list[str]:
    patterns = [
        r'"ou"\s*:\s*"(.*?)"',
        r'"imgurl"\s*:\s*"(.*?)"',
        r'data-iurl="(.*?)"',
        r"imgurl=([^&\"'<>]+)",
    ]
    found: list[str] = []
    for pattern in patterns:
        for match in re.findall(pattern, html, flags=re.DOTALL):
            url = normalize_image_url(clean_url(match))
            if not url.startswith(("http://", "https://")):
                continue
            if is_thumbnail_url(url):
                continue
            found.append(url)

    seen: set[str] = set()
    unique: list[str] = []
    for url in found:
        if url in seen:
            continue
        seen.add(url)
        unique.append(url)
    return unique


def extract_yandex_image_urls(html_text: str) -> list[str]:
    text = html.unescape(html_text)
    patterns = [
        r'"img_url"\s*:\s*"([^"]+)"',
        r"img_url=([^&\"'<>]+)",
    ]
    found: list[str] = []
    for pattern in patterns:
        for match in re.findall(pattern, text):
            url = clean_url(match)
            if not url.startswith(("http://", "https://")):
                continue
            if is_thumbnail_url(url):
                continue
            found.append(url)

    seen: set[str] = set()
    unique: list[str] = []
    for url in found:
        if url in seen:
            continue
        seen.add(url)
        unique.append(url)
    return unique


def extract_yandex_result_urls(html_text: str) -> list[str]:
    text = html.unescape(html_text)
    found: list[str] = []
    for match in re.findall(r'"img_href"\s*:\s*"([^"]+)"', text):
        url = clean_url(match)
        if not url.startswith(("http://", "https://")):
            continue
        found.append(url)

    seen: set[str] = set()
    unique: list[str] = []
    for url in found:
        if url in seen:
            continue
        seen.add(url)
        unique.append(url)
    return unique


def parse_page_index(url: str) -> int | None:
    parsed = urlparse(url)
    params = parse_qs(parsed.query)
    for key in ("p", "page", "pn"):
        value = params.get(key)
        if value:
            try:
                return int(value[0])
            except ValueError:
                return None
    return None


def extract_yandex_next_url(html_text: str, base_url: str, current_page: int) -> str | None:
    text = html.unescape(html_text)
    candidates: list[str] = []
    patterns = [
        r'"moreURL"\s*:\s*"([^"]+)"',
        r'"moreUrl"\s*:\s*"([^"]+)"',
        r'"more_url"\s*:\s*"([^"]+)"',
        r'"nextPage"\s*:\s*"([^"]+)"',
        r'"next"\s*:\s*"([^"]+)"',
        r'"url"\s*:\s*"([^"]+)"',
    ]
    for pattern in patterns:
        for match in re.findall(pattern, text):
            url = clean_url(match)
            url = urljoin(base_url, url)
            if "/images/search" not in url:
                continue
            if "img_url=" in url:
                continue
            if "text=" not in url:
                continue
            candidates.append(url)

    for match in re.findall(r"/images/search\\?[^\"'<>\\s]+", text):
        url = clean_url(match)
        url = urljoin(base_url, url)
        if "img_url=" in url:
            continue
        if "text=" not in url:
            continue
        if "p=" not in url and "page=" not in url:
            continue
        candidates.append(url)

    next_url = None
    next_page = None
    for url in candidates:
        page = parse_page_index(url)
        if page is None:
            continue
        if page <= current_page:
            continue
        if next_page is None or page < next_page:
            next_page = page
            next_url = url

    return next_url


def has_captcha_response(text: str) -> bool:
    lowered = text.lower()
    return "\"type\":\"captcha\"" in lowered or "\"type\": \"captcha\"" in lowered


def build_yandex_ajax_urls(search_url: str) -> list[str]:
    candidates = [
        add_query_param(search_url, "ajax", "1"),
        add_query_param(search_url, "format", "json"),
        add_query_param(add_query_param(search_url, "format", "json"), "ajax", "1"),
    ]
    seen: set[str] = set()
    unique: list[str] = []
    for url in candidates:
        if url in seen:
            continue
        seen.add(url)
        unique.append(url)
    return unique


def is_google_domain(netloc: str) -> bool:
    return (
        netloc.endswith("google.com")
        or netloc.endswith("googleusercontent.com")
        or netloc.endswith("gstatic.com")
        or netloc.endswith("googleadservices.com")
    )


def extract_result_urls(html: str) -> list[str]:
    found: list[str] = []
    for match in re.findall(r'href="/url\?q=([^&"]+)', html):
        url = clean_url(match)
        if not url.startswith(("http://", "https://")):
            continue
        parsed = urlparse(url)
        if not parsed.netloc or is_google_domain(parsed.netloc):
            continue
        if "webcache.googleusercontent.com" in url:
            continue
        found.append(url)

    seen: set[str] = set()
    unique: list[str] = []
    for url in found:
        if url in seen:
            continue
        seen.add(url)
        unique.append(url)
    return unique


def detect_provider(search_url: str) -> str:
    netloc = urlparse(search_url).netloc.lower()
    if "yandex." in netloc:
        return "yandex"
    if "google." in netloc:
        return "google"
    return "generic"


def extract_page_image_urls(html: str, base_url: str) -> list[str]:
    found: list[str] = []
    for tag in re.findall(r"<meta[^>]+>", html, flags=re.IGNORECASE):
        lower = tag.lower()
        if "og:image" in lower or "twitter:image" in lower:
            match = re.search(r'content=["\\\'](.*?)["\\\']', tag, flags=re.IGNORECASE)
            if not match:
                continue
            url = clean_url(match.group(1))
            url = urljoin(base_url, url)
            if not url.startswith(("http://", "https://")):
                continue
            if is_thumbnail_url(url):
                continue
            found.append(url)

    if not found:
        for tag in re.findall(r"<img[^>]+>", html, flags=re.IGNORECASE):
            for attr in ("data-src", "data-original", "data-lazy-src", "src"):
                match = re.search(
                    rf'{attr}=["\\\'](.*?)["\\\']', tag, flags=re.IGNORECASE
                )
                if not match:
                    continue
                url = clean_url(match.group(1))
                url = urljoin(base_url, url)
                if not url.startswith(("http://", "https://")):
                    continue
                if is_thumbnail_url(url):
                    continue
                if url.lower().endswith(".svg"):
                    continue
                found.append(url)
                break

    seen: set[str] = set()
    unique: list[str] = []
    for url in found:
        if url in seen:
            continue
        seen.add(url)
        unique.append(url)
    return unique


def find_start_index(output_dir: Path, prefix: str) -> int:
    pattern = re.compile(rf"{re.escape(prefix)}_(\d+)\.jpg$", re.IGNORECASE)
    max_index = 0
    for path in output_dir.glob(f"{prefix}_*.jpg"):
        match = pattern.match(path.name)
        if match:
            max_index = max(max_index, int(match.group(1)))
    return max_index + 1


def load_state(path: Path) -> set[str]:
    if not path.exists():
        return set()
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            url = line.strip()
            if url:
                seen.add(url)
    return seen


def append_state(handle, url: str) -> None:
    handle.write(url)
    handle.write("\n")
    handle.flush()


def save_image(
    content: bytes, output_path: Path, max_dim: int, resized_counter: list[int]
) -> bool:
    try:
        with Image.open(BytesIO(content)) as img:
            img = ImageOps.exif_transpose(img)
            width, height = img.size
            max_side = max(width, height)
            if max_dim > 0 and max_side > max_dim:
                scale = max_side / max_dim
                new_size = (int(round(width / scale)), int(round(height / scale)))
                img = img.resize(new_size, Image.LANCZOS)
                resized_counter[0] += 1

            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            img.save(output_path, format="JPEG", quality=95)
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def download_image_url(
    session: requests.Session,
    url: str,
    output_path: Path,
    max_dim: int,
    timeout: float,
    resized_counter: list[int],
) -> str | None:
    try:
        image_response = session.get(url, timeout=timeout)
    except requests.RequestException:
        return "failed"

    if image_response.status_code != 200:
        return "failed"

    content_type = image_response.headers.get("Content-Type", "").lower()
    if "image" not in content_type:
        return "non_image"

    saved = save_image(image_response.content, output_path, max_dim, resized_counter)
    if saved:
        return None
    return "failed"


def fetch_search_results(
    session: requests.Session,
    search_url: str,
    timeout: float,
    user_agent: str,
    provider: str,
    current_page: int,
    yandex_captcha_warned: list[bool] | None,
) -> tuple[list[str], list[str], str | None]:
    headers = {
        "User-Agent": user_agent,
        "Referer": search_url,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    response = session.get(search_url, headers=headers, timeout=timeout)
    response.raise_for_status()
    if provider == "google" and "consent.google.com" in str(response.url).lower():
        print("Warning: Google returned a consent page. The URL may need manual access.")

    next_url = None
    if provider == "yandex":
        image_urls = extract_yandex_image_urls(response.text)
        result_urls = extract_yandex_result_urls(response.text)
        next_url = extract_yandex_next_url(response.text, search_url, current_page)
    else:
        image_urls = extract_image_urls(response.text)
        result_urls = extract_result_urls(response.text)

    if not image_urls and not result_urls:
        fallback_headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(search_url, headers=fallback_headers, timeout=timeout)
        response.raise_for_status()
        if provider == "yandex":
            image_urls = extract_yandex_image_urls(response.text)
            result_urls = extract_yandex_result_urls(response.text)
            next_url = extract_yandex_next_url(response.text, search_url, current_page)
        else:
            image_urls = extract_image_urls(response.text)
            result_urls = extract_result_urls(response.text)

    if provider == "yandex" and not image_urls and not result_urls:
        session.get("https://yandex.ru/", headers=headers, timeout=timeout)
        response = session.get(search_url, headers=headers, timeout=timeout)
        response.raise_for_status()
        image_urls = extract_yandex_image_urls(response.text)
        result_urls = extract_yandex_result_urls(response.text)
        next_url = extract_yandex_next_url(response.text, search_url, current_page)

    if provider == "yandex" and not image_urls and not result_urls:
        for ajax_url in build_yandex_ajax_urls(search_url):
            ajax_headers = {
                **headers,
                "Accept": "application/json, text/plain, */*",
                "X-Requested-With": "XMLHttpRequest",
                "Origin": "https://yandex.ru",
            }
            response = session.get(ajax_url, headers=ajax_headers, timeout=timeout)
            response.raise_for_status()
            if has_captcha_response(response.text):
                if yandex_captcha_warned is not None and not yandex_captcha_warned[0]:
                    print(
                        "Warning: Yandex returned a captcha response for the AJAX endpoint."
                    )
                    yandex_captcha_warned[0] = True
                continue
            image_urls = extract_yandex_image_urls(response.text)
            result_urls = extract_yandex_result_urls(response.text)
            next_url = extract_yandex_next_url(response.text, search_url, current_page)
            if image_urls or result_urls:
                break

    if provider == "google" and not image_urls and not result_urls and "gbv=1" not in search_url:
        fallback_headers = {"User-Agent": "Mozilla/5.0"}
        fallback_url = add_query_param(search_url, "gbv", "1")
        response = requests.get(fallback_url, headers=fallback_headers, timeout=timeout)
        response.raise_for_status()
        image_urls = extract_image_urls(response.text)
        result_urls = extract_result_urls(response.text)

    return image_urls, result_urls, next_url


def download_google_images(
    search_url: str,
    output_dir: Path,
    count: int,
    max_dim: int,
    prefix: str,
    timeout: float,
    delay: float,
    user_agent: str,
    state_path: Path | None,
    max_pages: int,
    page_size: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    provider = detect_provider(search_url)

    print(
        "Starting image download.\n"
        f"- Provider: {provider}\n"
        f"- Search URL: {search_url}\n"
        f"- Output dir: {output_dir}\n"
        f"- Target count: {count}\n"
        f"- Max dimension: {max_dim if max_dim > 0 else 'disabled'}\n"
        f"- Filename prefix: {prefix}\n"
        f"- Timeout: {timeout}s\n"
        f"- Delay: {delay}s\n"
        f"- Max pages: {'unlimited' if max_pages <= 0 else max_pages}\n"
        f"- Page size: {page_size if provider == 'google' else 'n/a'}"
    )

    session = requests.Session()

    downloaded = 0
    failed = 0
    skipped = 0
    non_image = 0
    page_attempted = 0
    page_failed = 0
    page_no_image = 0
    attempted = 0
    resized_counter = [0]
    seen: set[str] = set()
    next_index = find_start_index(output_dir, prefix)
    downloaded_urls = load_state(state_path) if state_path else set()
    state_written = 0
    search_pages = 0
    empty_pages = 0

    if state_path:
        print(f"- State file: {state_path} (loaded {len(downloaded_urls)} URLs)")

    start_time = time.time()

    state_handle = None
    if state_path:
        state_handle = state_path.open("a", encoding="utf-8")

    try:
        yandex_captcha_warned = [False]
        page_index = 0
        next_page_url: str | None = None
        while downloaded < count:
            if max_pages > 0 and page_index >= max_pages:
                break

            if provider == "yandex":
                if next_page_url:
                    page_url = next_page_url
                else:
                    page_url = add_query_param(search_url, "p", str(page_index))
            else:
                page_url = add_query_param(
                    search_url, "start", str(page_index * page_size)
                )
                page_url = add_query_param(page_url, "num", str(page_size))
            image_urls, result_urls, next_page_url = fetch_search_results(
                session,
                page_url,
                timeout,
                user_agent,
                provider,
                page_index,
                yandex_captcha_warned if provider == "yandex" else None,
            )
            search_pages += 1

            if not image_urls and not result_urls:
                empty_pages += 1
                if empty_pages >= 2:
                    break
            else:
                empty_pages = 0

            for url in image_urls:
                if downloaded >= count:
                    break
                url = normalize_image_url(url)
                if url in downloaded_urls:
                    skipped += 1
                    continue
                if url in seen:
                    skipped += 1
                    continue
                seen.add(url)

                attempted += 1
                tail = url.split("?")[0]
                if len(tail) > 55:
                    tail = tail[-55:]

                output_path = output_dir / f"{prefix}_{next_index:04d}.jpg"
                result = download_image_url(
                    session, url, output_path, max_dim, timeout, resized_counter
                )
                if result is None:
                    downloaded += 1
                    next_index += 1
                    downloaded_urls.add(url)
                    if state_handle:
                        append_state(state_handle, url)
                        state_written += 1
                else:
                    if result == "non_image":
                        non_image += 1
                    else:
                        failed += 1

                elapsed = time.time() - start_time
                rate = downloaded / elapsed if downloaded else 0.0
                eta = format_eta((count - downloaded) / rate) if rate else format_eta(None)
                render_progress(downloaded, count, eta, tail)

                if delay > 0:
                    time.sleep(delay)

            for page_link in result_urls:
                if downloaded >= count:
                    break
                if page_link in seen:
                    skipped += 1
                    continue
                seen.add(page_link)
                page_attempted += 1

                tail = page_link.split("?")[0]
                if len(tail) > 55:
                    tail = tail[-55:]

                try:
                    page_response = session.get(page_link, timeout=timeout)
                    page_response.raise_for_status()
                except requests.RequestException:
                    page_failed += 1
                    elapsed = time.time() - start_time
                    rate = downloaded / elapsed if downloaded else 0.0
                    eta = format_eta((count - downloaded) / rate) if rate else format_eta(None)
                    render_progress(downloaded, count, eta, tail)
                    continue

                candidates = extract_page_image_urls(page_response.text, page_link)
                if not candidates:
                    page_no_image += 1
                    elapsed = time.time() - start_time
                    rate = downloaded / elapsed if downloaded else 0.0
                    eta = format_eta((count - downloaded) / rate) if rate else format_eta(None)
                    render_progress(downloaded, count, eta, tail)
                    continue

                for img_url in candidates:
                    if downloaded >= count:
                        break
                    if img_url in downloaded_urls:
                        skipped += 1
                        continue
                    if img_url in seen:
                        skipped += 1
                        continue
                    seen.add(img_url)
                    attempted += 1
                    output_path = output_dir / f"{prefix}_{next_index:04d}.jpg"
                    result = download_image_url(
                        session, img_url, output_path, max_dim, timeout, resized_counter
                    )
                    if result is None:
                        downloaded += 1
                        next_index += 1
                        downloaded_urls.add(img_url)
                        if state_handle:
                            append_state(state_handle, img_url)
                            state_written += 1
                        break
                    if result == "non_image":
                        non_image += 1
                    else:
                        failed += 1

                elapsed = time.time() - start_time
                rate = downloaded / elapsed if downloaded else 0.0
                eta = format_eta((count - downloaded) / rate) if rate else format_eta(None)
                render_progress(downloaded, count, eta, tail)

                if delay > 0:
                    time.sleep(delay)

            if provider == "yandex":
                next_page_index = (
                    parse_page_index(next_page_url) if next_page_url else None
                )
                if next_page_index is None or next_page_index <= page_index:
                    page_index += 1
                else:
                    page_index = next_page_index
            else:
                page_index += 1
    finally:
        if state_handle:
            state_handle.close()

    if count > 0:
        print()

    if downloaded < count:
        print(
            f"Downloaded {downloaded}/{count} images. "
            "Ran out of usable URLs or some downloads failed."
        )
    else:
        print(f"Downloaded {downloaded} images.")

    print(
        "Stats: "
        f"attempted={attempted}, failed={failed}, skipped={skipped}, "
        f"non_image={non_image}, resized={resized_counter[0]}, "
        f"pages={page_attempted}, page_failed={page_failed}, "
        f"page_no_image={page_no_image}, state_written={state_written}, "
        f"search_pages={search_pages}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download full-resolution images from Google or Yandex Images."
    )
    parser.add_argument(
        "--url",
        required=True,
        help="Google or Yandex Images search URL to scrape.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of images to download. Default: 100",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (relative to CWD). Default: './images'",
    )
    parser.add_argument(
        "--max-dim",
        type=int,
        default=1024,
        help="Resize images so the longest side is <= this value. Set 0 to disable.",
    )
    parser.add_argument(
        "--prefix",
        default="google",
        help="Filename prefix for saved images. Default: 'google'",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="Request timeout in seconds. Default: 15",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="Delay between downloads in seconds. Default: 0.1",
    )
    parser.add_argument(
        "--user-agent",
        default=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        help="User-Agent header for requests.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=0,
        help="Maximum number of search result pages to fetch. 0 means unlimited.",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=20,
        help="Number of results per page for pagination (Google only). Default: 20",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=None,
        help=(
            "Path to a text file for tracking downloaded image URLs. "
            "Defaults to '<output>/<prefix>_downloaded.txt'."
        ),
    )
    parser.add_argument(
        "--no-state",
        action="store_true",
        help="Disable URL tracking state file.",
    )

    args = parser.parse_args()

    cwd = Path.cwd()
    script_dir = Path(__file__).resolve().parent
    output_dir = (
        resolve_path(args.output, cwd) if args.output is not None else cwd / "images"
    )
    if args.no_state:
        state_path = None
    else:
        state_path = (
            resolve_path(args.state_file, cwd)
            if args.state_file is not None
            else output_dir / f"{args.prefix}_downloaded.txt"
        )

    print(f"Data root (CWD): {cwd}")
    print(f"Script dir: {script_dir}")

    if args.count <= 0:
        raise ValueError("--count must be greater than zero.")

    warnings.filterwarnings("ignore", category=UserWarning, module="PIL")

    download_google_images(
        search_url=args.url,
        output_dir=output_dir,
        count=args.count,
        max_dim=args.max_dim,
        prefix=args.prefix,
        timeout=args.timeout,
        delay=max(args.delay, 0.0),
        user_agent=args.user_agent,
        state_path=state_path,
        max_pages=args.max_pages,
        page_size=max(args.page_size, 1),
    )


if __name__ == "__main__":
    main()
