"""Create an explicitly development-only workflow fixture, never a test demo."""

import argparse
import hashlib
import zipfile
from pathlib import Path

from secom_quality.common import sha256_file, write_json
from secom_quality.vision.data import load_development


def build(root: Path):
    frame = load_development(root)
    # Balanced only for illustration; it has no benchmark prevalence or AP claim.
    chosen = frame.sort_values("record_id").groupby("label", sort=True).head(10)
    output = root / "vision/demo"
    output.mkdir(parents=True, exist_ok=True)
    archive = output / "development_images.zip"
    records = []
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        for row in chosen.itertuples():
            identity = "demo-" + hashlib.sha256(row.record_id.encode()).hexdigest()[:12]
            member = identity + ".png"
            info = zipfile.ZipInfo(member, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            zipped.writestr(info, Path(row.image_path).read_bytes())
            records.append(
                {
                    "demo_id": identity,
                    "source_record_id": row.record_id,
                    "source_partition": "development",
                    "sha256": sha256_file(Path(row.image_path)),
                }
            )
    write_json(
        output / "manifest.json",
        {
            "purpose": "Workflow illustration only; deliberately balanced development images overlap model training.",
            "performance_claim": "None. This fixture cannot estimate generalisation or factory prevalence.",
            "dataset": "KolektorSDD2",
            "license": "CC BY-NC-SA 4.0",
            "source": "https://www.vicos.si/resources/kolektorsdd2/",
            "records": records,
            "archive_sha256": sha256_file(archive),
            "test_images_included": False,
        },
    )
    print(archive)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("workspace"))
    build(parser.parse_args().workspace)
