"""Record a five-minute captioned walkthrough of the real local application."""

import argparse
import json
import subprocess
import time
from pathlib import Path

import httpx
from playwright.sync_api import sync_playwright

from secom_quality.common import sha256_file, utc_now, write_json

CHAPTERS = [
    (
        "Overview",
        35,
        "A complete retrospective research system. The frozen holdout result is no observed improvement: 2 of 26 failures caught in 47 reviews, versus 2.59 expected randomly.",
    ),
    (
        "Data audit",
        75,
        "The verified archive contains 1,567 observations and 590 anonymous measurements. Labels, timestamps and identities never become model features. The later 471 records are excluded from development.",
    ),
    (
        "Experiments",
        120,
        "The model grid is fixed: 16 eligible configurations and two diagnostic configurations, each evaluated on two forward folds. The selected 15-leaf boosting model stays frozen after final evaluation.",
    ),
    (
        "Inspection queue",
        185,
        "This is a development-only demonstration batch. The same verified predictor serves the CLI and API. Top-k inspection is a closed-batch capacity policy; risk scores are not calibrated probabilities.",
    ),
    (
        "Monitoring",
        255,
        "Replay has its own early-development model and a disjoint development stream. Outcomes arrive two virtual batches after predictions. Synthetic stress demonstrates monitoring mechanics, not factory drift validation.",
    ),
    (
        "Model card",
        300,
        "Engineering verification and scientific findings are separate. This release preserves its negative result, code and artifact fingerprints, research limitations and reproducible local workflow.",
    ),
]


def record(base_url: str, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "secom_quality_lab_walkthrough.webm"
    if destination.exists():
        raise FileExistsError(
            "Preserve existing footage; choose a new output directory for a new recording"
        )
    errors = []
    with httpx.Client(base_url=base_url, timeout=30) as client:
        ready = client.get("/health/ready")
        ready.raise_for_status()
        sample = client.get("/api/v1/demo/inputs")
        sample.raise_for_status()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        context = browser.new_context(
            viewport={"width": 1280, "height": 800},
            record_video_dir=str(output / "recording"),
            record_video_size={"width": 1280, "height": 800},
            reduced_motion="reduce",
        )
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        video = page.video
        started = time.monotonic()
        page.goto(base_url)
        page.get_by_role("heading", name="Quality, with evidence.").wait_for()

        def caption(title: str, text: str) -> None:
            page.evaluate(
                """({title,text}) => {
                let box=document.getElementById('walkthrough-caption');
                if (!box) {box=document.createElement('div');box.id='walkthrough-caption';document.body.append(box);}
                Object.assign(box.style,{position:'fixed',bottom:'16px',left:'252px',right:'22px',padding:'15px 19px',background:'#17362ff5',color:'white',borderRadius:'9px',font:'15px/1.55 system-ui',zIndex:'10000',boxShadow:'0 5px 25px #0003',pointerEvents:'none'});
                box.replaceChildren();const strong=document.createElement('strong');strong.textContent=title+' — ';box.append(strong,document.createTextNode(text));
            }""",
                {"title": title, "text": text},
            )

        def hold_until(seconds: int) -> None:
            while time.monotonic() - started < seconds:
                remaining = seconds - (time.monotonic() - started)
                page.wait_for_timeout(min(3000, max(1, int(remaining * 1000))))

        for title, end, text in CHAPTERS:
            if title != "Overview":
                page.get_by_role("button", name=title, exact=True).click()
            caption(title, text)
            if title == "Data audit":
                page.wait_for_timeout(2500)
                page.mouse.wheel(0, 480)
            elif title == "Experiments":
                page.wait_for_timeout(2500)
                page.mouse.wheel(0, 350)
            elif title == "Inspection queue":
                page.get_by_label("Choose CSV file").set_input_files(
                    {
                        "name": "approved_development.csv",
                        "mimeType": "text/csv",
                        "buffer": sample.content,
                    }
                )
                page.get_by_role("button", name="Validate and score", exact=True).click()
                page.get_by_role("heading", name="Prioritised records").wait_for()
                page.get_by_role("heading", name="Prioritised records").scroll_into_view_if_needed()
                page.wait_for_timeout(4000)
                page.get_by_role("button", name="Inspect secom-r000551", exact=True).click()
                page.wait_for_timeout(8000)
                page.get_by_role("button", name="Close record details", exact=True).click()
                with page.expect_download() as download:
                    page.get_by_role("link", name="Export queue", exact=True).click()
                download.value.save_as(str(output / "demonstration_inspection_queue.csv"))
            elif title == "Monitoring":
                page.get_by_role("button", name="Missingness stress", exact=False).click()
                page.get_by_role("button", name="New replay", exact=True).click()
                for _ in range(7):
                    page.get_by_role("button", name="Step 20", exact=True).click()
                    page.wait_for_timeout(600)
                page.mouse.wheel(0, 450)
                page.wait_for_timeout(8000)
                page.mouse.wheel(0, -650)
                page.get_by_role("button", name="Play", exact=True).click()
                page.wait_for_timeout(4000)
                page.get_by_role("button", name="Pause", exact=True).click()
            page.screenshot(path=str(output / f"{title.lower().replace(' ', '_')}.png"))
            hold_until(end)
        context.close()
        assert video is not None
        video.save_as(str(destination))
        media = browser.new_page()

        def duration(path: Path) -> float:
            media.goto(path.resolve().as_uri())
            media.wait_for_function(
                "document.querySelector('video') && document.querySelector('video').readyState >= 1"
            )
            return float(media.locator("video").evaluate("video => video.duration"))

        source_duration = duration(destination)
        if abs(source_duration - 300) > 0.5:
            runtime = Path(playwright.chromium.executable_path).parents[2]
            encoders = sorted(runtime.glob("ffmpeg-*/ffmpeg*"))
            if not encoders:
                raise RuntimeError(
                    "Playwright's video encoder is unavailable; reinstall its browser runtime"
                )
            normalized = output / "normalized.webm"
            subprocess.run(
                [
                    str(encoders[-1]),
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-n",
                    "-itsscale",
                    str(300 / source_duration),
                    "-i",
                    str(destination),
                    "-map",
                    "0:v:0",
                    "-c",
                    "copy",
                    str(normalized),
                ],
                check=True,
            )
            destination.replace(output / "recording/raw_capture.webm")
            normalized.replace(destination)
        media_duration = duration(destination)
        if not 295 <= media_duration <= 305:
            raise RuntimeError("Recorded video did not meet the five-minute duration target")
        browser.close()
    if errors:
        raise RuntimeError(f"Browser errors during recorded demonstration: {errors}")
    manifest = {
        "created_at": utc_now(),
        "workflow_elapsed_seconds": time.monotonic() - started,
        "source_capture_duration_seconds": source_duration,
        "media_duration_seconds": media_duration,
        "playback_note": "Presentation timestamps normalized to approximately five minutes when necessary; captured frames and results are unchanged.",
        "chapter_time_basis": "Approximate script schedule",
        "file": destination.name,
        "sha256": sha256_file(destination),
        "bundle_id": ready.json()["bundle_id"],
        "browser_errors": errors,
        "format": "1280x800 WebM, real local Chromium capture, explanatory captions, no voice track",
        "chapters": [
            {"title": title, "ends_at_seconds": end, "caption": text}
            for title, end, text in CHAPTERS
        ],
    }
    write_json(output / "recording_manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--output", type=Path, default=Path("evidence/demo"))
    args = parser.parse_args()
    print(json.dumps(record(args.base_url, args.output), indent=2))
