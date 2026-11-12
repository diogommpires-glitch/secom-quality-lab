"""Stream the official AD2 archive, with Range resume and no test-image decoding."""

import argparse
import hashlib
import json
import os
import re
import time
import urllib.request
from pathlib import Path

PAGE = "https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2/downloads"


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def fetch(destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "mvtec_ad_2.tar.gz"
    completed = destination / "download.json"
    if archive.exists() and completed.exists():
        manifest = json.loads(completed.read_text())
        with archive.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != manifest["sha256"]:
            raise ValueError("Existing AD2 archive checksum differs from its acquisition record")
        return manifest
    with urllib.request.urlopen(PAGE, timeout=60) as response:
        html = response.read()
    links = re.findall(r'href=["\x27]([^"\x27]+mvtec_ad_2\.tar\.gz)["\x27]', html.decode())
    if len(set(links)) != 1 or not links[0].startswith("https://www.mydrive.ch/shares/"):
        raise ValueError("Expected one official HTTPS AD2 archive link")
    url = links[0]
    partial = archive.with_suffix(".gz.partial")
    position = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "SECOM-Quality-Lab research data acquisition"}
    if position:
        headers["Range"] = f"bytes={position}-"
    request = urllib.request.Request(url, headers=headers)
    digest = hashlib.sha256()
    if position:
        with partial.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
    start, updated = time.monotonic(), 0.0
    with urllib.request.urlopen(request, timeout=120) as response:
        if position and (
            response.status != 206
            or not response.headers.get("Content-Range", "").startswith(f"bytes {position}-")
        ):
            raise ValueError("Download endpoint did not honour exact Range resume")
        expected = int(response.headers.get("Content-Length", 0)) + position
        metadata = {
            "source_page": PAGE,
            "url": url,
            "page_sha256": hashlib.sha256(html).hexdigest(),
            "expected_bytes": expected,
            "sha256_policy": "First official acquisition pinned locally; publisher does not supply a checksum here.",
        }
        with partial.open("ab" if position else "wb") as output:
            while chunk := response.read(2 * 1024 * 1024):
                if position == 0 and chunk[:2] != b"\x1f\x8b":
                    raise ValueError("Official endpoint did not return a gzip archive")
                output.write(chunk)
                digest.update(chunk)
                position += len(chunk)
                if time.monotonic() - updated > 10:
                    write_json(
                        destination / "progress.json",
                        {
                            **metadata,
                            "state": "downloading",
                            "received_bytes": position,
                            "elapsed_seconds": time.monotonic() - start,
                        },
                    )
                    updated = time.monotonic()
        if expected and position != expected:
            raise ValueError("Incomplete AD2 transfer; partial archive retained for resume")
    manifest = {
        **metadata,
        "received_bytes": position,
        "sha256": digest.hexdigest(),
        "state": "complete",
    }
    partial.rename(archive)
    write_json(completed, manifest)
    write_json(destination / "progress.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(fetch(arguments.destination), indent=2))
