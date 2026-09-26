"""HTTP download helpers with retries and progress bars."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Mapping

import requests
from tqdm import tqdm

from track1_data_engine.core import get_logger

logger = get_logger()


class DownloadError(RuntimeError):
    """Raised when a remote file cannot be retrieved."""


def http_get_to_file(
    url: str,
    dest: Path,
    *,
    timeout: int = 120,
    max_retries: int = 5,
    headers: Mapping[str, str] | None = None,
    skip_existing: bool = True,
    min_bytes: int = 1024,
) -> Path:
    """Download `url` to `dest` with exponential backoff."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if skip_existing and dest.exists() and dest.stat().st_size >= min_bytes:
        logger.info("Skipping existing file %s", dest)
        return dest

    last_error: Exception | None = None
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, max_retries + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout, headers=headers) as response:
                response.raise_for_status()
                total = int(response.headers.get("Content-Length", 0))
                with tmp.open("wb") as handle, tqdm(
                    total=total or None,
                    unit="B",
                    unit_scale=True,
                    desc=dest.name,
                    leave=False,
                ) as bar:
                    for chunk in response.iter_content(chunk_size=1 << 16):
                        if chunk:
                            handle.write(chunk)
                            bar.update(len(chunk))
            if tmp.stat().st_size < min_bytes:
                raise DownloadError(f"{url} produced undersized file ({tmp.stat().st_size} bytes).")
            tmp.replace(dest)
            logger.info("Downloaded %s -> %s (%s bytes)", url, dest, dest.stat().st_size)
            return dest
        except (requests.RequestException, OSError, DownloadError) as exc:
            last_error = exc
            logger.warning("Attempt %s/%s failed for %s: %s", attempt, max_retries, url, exc)
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            time.sleep(min(2**attempt, 30))
    raise DownloadError(f"Failed to download {url} after {max_retries} attempts.") from last_error
