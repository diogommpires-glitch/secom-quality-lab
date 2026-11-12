"""One resumable local research run; never hides failed phases or test exposure."""

import argparse
import os
import traceback
from pathlib import Path

from secom_quality.common import read_json, utc_now, write_json

from .study import exclusive


def run(root: Path, device="cuda"):
    from .bundle import build_bundle
    from .data import audit_data, fetch_data
    from .protocol import checked_protocol, freeze_protocol
    from .report import build_report
    from .robustness import run_stress
    from .study import evaluate_final, freeze_selection, run_experiments

    root = Path(root).resolve()
    runtime = root / "vision/runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    state = {"pid": os.getpid(), "started_at": utc_now(), "workspace": str(root), "device": device}

    def status(phase, outcome="running", **extra):
        state.update(phase=phase, state=outcome, updated_at=utc_now(), **extra)
        write_json(runtime / "run.json", state)
        print(f"{utc_now()} {phase}: {outcome}", flush=True)

    with exclusive(runtime / "runner.lock"):
        try:
            status("source_verification")
            fetch_data(root)
            audit_data(root)
            status("protocol_freeze")
            path = root / "vision/research/protocol.json"
            protocol = checked_protocol(root) if path.exists() else freeze_protocol(root)
            status("development_experiments", protocol_hash=protocol["protocol_hash"])
            run_experiments(root, device=device)
            build_report(root)
            status("development_corruption_analysis")
            run_stress(root, device=device)
            status("selection_freeze")
            selection = freeze_selection(root)
            status("final_training_and_evaluation", selection_hash=selection["selection_hash"])
            evaluate_final(root, device=device)
            status("published_report")
            build_report(root)
            status("serving_bundle")
            bundle = build_bundle(root)
            status(
                "primary_study",
                "completed",
                bundle_id=bundle["bundle_id"],
                ad2_status="pending_independent_challenge",
            )
            archive = root / "vision/ad2_downloads/mvtec_ad_2.tar.gz"
            source_manifest = root / "vision/ad2/source/manifest.json"
            if archive.exists() or source_manifest.exists():
                from .ad2 import audit_ad2, export_ad2, freeze_ad2, run_ad2

                if not source_manifest.exists():
                    status("ad2_source_audit")
                    audit_ad2(root, archive)
                status("ad2_protocol_freeze")
                freeze_ad2(root)
                status("ad2_public_challenge")
                run_ad2(root, device=device)
                status("ad2_private_export")
                export_ad2(root, device=device)
                status(
                    "research", "completed", ad2_status="local_export_ready_no_external_submission"
                )
            return read_json(runtime / "run.json")
        except BaseException as exc:
            status(
                state.get("phase", "startup"),
                "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                error_type=type(exc).__name__,
                error=str(exc),
                traceback=traceback.format_exc(),
            )
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()
    run(args.workspace, args.device)


if __name__ == "__main__":
    main()
