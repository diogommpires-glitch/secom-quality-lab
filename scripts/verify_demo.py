"""Real-bundle HTTP checks, local benchmarks and browser screenshots."""

import argparse
import json
import platform
import time
from pathlib import Path

import httpx
import numpy as np
import psutil
from playwright.sync_api import sync_playwright

from secom_quality.artifacts import load_bundle
from secom_quality.common import write_json
from secom_quality.inference import parse_csv, records_csv, score_records


def verify(base_url: str, root: Path, output: Path, server_pid: int | None) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    with httpx.Client(base_url=base_url, timeout=30) as client:
        ready = client.get("/health/ready")
        ready.raise_for_status()
        assert ready.json()["monitoring_ready"] is True
        sample = client.get("/api/v1/demo/inputs")
        sample.raise_for_status()
        records = parse_csv(sample.content)
        bundle = load_bundle(root / "bundles/main")
        expected = score_records(records, bundle)
        latencies = []
        for record, prediction in zip(records, expected, strict=True):
            sensors = {f"sensor_{i:03d}": value for i, value in enumerate(record["values"])}
            start = time.perf_counter()
            response = client.post(
                "/api/v1/predictions", json={"record_id": record["record_id"], "sensors": sensors}
            )
            latencies.append(time.perf_counter() - start)
            response.raise_for_status()
            assert response.json()["risk_score"] == prediction["risk_score"]
            assert response.json()["calibration_status"] == "not_validated"
        expanded = [
            {**records[i % len(records)], "record_id": f"benchmark-{i}"} for i in range(100)
        ]
        content = records_csv(expanded)
        start = time.perf_counter()
        batch = client.post(
            "/api/v1/batches",
            data={"client_request_id": "release-benchmark-100"},
            files={"file": ("benchmark.csv", content, "text/csv")},
        )
        batch_seconds = time.perf_counter() - start
        batch.raise_for_status()
        result = batch.json()
        assert result["review_capacity"] == 10
        assert sum(row["review_recommended"] for row in result["predictions"]) == 10
        retry = client.post(
            "/api/v1/batches",
            data={"client_request_id": "release-benchmark-100"},
            files={"file": ("benchmark.csv", content, "text/csv")},
        )
        retry.raise_for_status()
        assert retry.json()["batch_id"] == result["batch_id"]
        assert retry.json()["replayed"] is True
        invalid = client.post(
            "/api/v1/batches",
            data={"client_request_id": "invalid-release-input"},
            files={"file": ("invalid.csv", b"record_id,sensor_000\nbroken,infinity\n", "text/csv")},
        )
        assert invalid.status_code == 422
        assert invalid.json()["detail"]["request_id"] == invalid.headers["x-request-id"]
        protected = client.post(
            "/api/v1/predictions",
            json={
                "record_id": "secom-r001097",
                "sensors": {
                    f"sensor_{i:03d}": value for i, value in enumerate(records[0]["values"])
                },
            },
        )
        assert protected.status_code == 403

    browser_errors: list[str] = []
    external_requests: list[str] = []
    screenshots = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
        def local_only(route):
            if route.request.url.startswith(base_url.rstrip("/") + "/"):
                route.continue_()
            else:
                external_requests.append(route.request.url)
                route.abort()

        page.route("**/*", local_only)
        page.on("pageerror", lambda error: browser_errors.append(str(error)))
        page.goto(base_url)
        page.get_by_role("heading", name="Quality, with evidence.").wait_for()
        for nav, name in [
            (None, "overview"),
            ("Data audit", "data_audit"),
            ("Experiments", "experiments"),
            ("Inspection queue", "inspection"),
            ("Monitoring", "monitoring"),
            ("Model card", "model_card"),
        ]:
            if nav:
                page.get_by_role("button", name=nav, exact=True).click()
            if name == "inspection":
                page.get_by_label("Choose CSV file").set_input_files(
                    {"name": "development.csv", "mimeType": "text/csv", "buffer": sample.content}
                )
                page.get_by_role("button", name="Validate and score", exact=True).click()
                page.get_by_role("heading", name="Prioritised records").wait_for()
            if name == "monitoring":
                page.get_by_role("button", name="New replay", exact=True).click()
                for _ in range(7):
                    page.get_by_role("button", name="Step 20", exact=True).click()
                    page.wait_for_timeout(100)
            page.wait_for_timeout(500)
            destination = output / f"{name}.png"
            page.screenshot(path=str(destination), full_page=True)
            screenshots.append(destination.name)
            assert page.locator("body").evaluate("element => element.scrollWidth <= innerWidth"), (
                f"Horizontal overflow: {name}"
            )
        page.set_viewport_size({"width": 390, "height": 844})
        page.get_by_role("button", name="Overview", exact=True).click()
        page.wait_for_timeout(400)
        page.screenshot(path=str(output / "overview_mobile.png"), full_page=True)
        assert page.locator("body").evaluate("element => element.scrollWidth <= innerWidth"), (
            "Mobile overflow"
        )
        browser.close()
    assert not browser_errors, browser_errors
    assert not external_requests, external_requests
    evidence = {
        "status": "passed",
        "base_url": base_url,
        "bundle_id": bundle.manifest["bundle_id"],
        "python": platform.python_version(),
        "platform": platform.platform(),
        "logical_cpus": psutil.cpu_count(),
        "memory_gib": round(psutil.virtual_memory().total / 2**30, 2),
        "single_record_requests": len(latencies),
        "single_record_p50_ms": float(np.median(latencies) * 1000),
        "single_record_p95_ms": float(np.quantile(latencies, 0.95) * 1000),
        "batch_records": 100,
        "batch_seconds": batch_seconds,
        "server_rss_mib": round(psutil.Process(server_pid).memory_info().rss / 2**20, 2)
        if server_pid
        else None,
        "parity": "HTTP scores exactly match verified-bundle core scores for 20 development records",
        "browser_errors": browser_errors,
        "external_requests": external_requests,
        "offline_browser": "All nonlocal requests were blocked throughout the browser workflows.",
        "screenshots": screenshots,
        "limitations": "Local small-workload measurements on this machine; not a deployment or throughput guarantee. Benchmark rows repeat development measurements under unique record IDs.",
    }
    write_json(output / "verification.json", evidence)
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--workspace", type=Path, default=Path("workspace"))
    parser.add_argument("--output", type=Path, default=Path("evidence/verification"))
    parser.add_argument("--server-pid", type=int)
    args = parser.parse_args()
    print(json.dumps(verify(args.base_url, args.workspace, args.output, args.server_pid), indent=2))
