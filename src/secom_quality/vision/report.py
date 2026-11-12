"""Publish truthful aggregates and figures without copying evaluation images."""

from pathlib import Path

import numpy as np
import pandas as pd

from secom_quality.common import read_json, write_json

from .study import checked_protocol, comparison


def build_report(root: Path):
    root = Path(root)
    protocol = checked_protocol(root)
    research = root / "vision/research"
    output = root / "vision/reports"
    output.mkdir(parents=True, exist_ok=True)
    rows = comparison(root)
    table = [
        {
            "model": row["config"]["config_id"],
            "valid": row["valid"],
            "mean_recall_at_10pct": row["mean_recall"],
            "mean_image_ap": row["mean_ap"],
            "mean_validation_seconds": row["mean_inference_seconds"],
            "completed_fits": row["completed_fits"],
        }
        for row in rows
    ]
    pd.DataFrame(table).to_csv(output / "development.csv", index=False)
    lines = [
        "# Industrial Vision Study",
        "",
        "Retrospective visual inspection research, separate from the frozen SECOM study.",
        "",
        f"Protocol: `{protocol['protocol_hash']}`",
        "",
        "## Development",
        "",
        "| Configuration | Valid | Mean recall at 10% | Mean image AP |",
        "|---|---|---:|---:|",
    ]
    for row in table:
        recall = f"{row['mean_recall_at_10pct']:.4f}" if row["valid"] else "pending/invalid"
        ap = f"{row['mean_image_ap']:.4f}" if row["valid"] else "pending/invalid"
        lines.append(f"| {row['model']} | {row['valid']} | {recall} | {ap} |")
    final_path = research / "final/report.json"
    if final_path.exists():
        final = read_json(final_path)
        write_json(output / "published-report.json", final)
        lines += [
            "",
            "## Frozen final evaluation",
            "",
            f"Scientific status: **{final['scientific_status']}**.",
            "",
            f"Serving model: `{final['model_id']}`; seed `{final['serving_seed']}`.",
            "",
            "| Seed | Recall at 10% | Image AP | Failures caught |",
            "|---|---:|---:|---:|",
        ]
        for run in final["per_seed"]:
            m = run["metrics"]
            lines.append(
                f"| {run['seed']} | {m['primary']['recall']:.4f} | {m['average_precision']:.4f} | {m['primary']['true_positives']} |"
            )
        lines += [
            "",
            f"One-sided random-inspection comparison: p={final['primary']['random_review_one_sided_pvalue']:.6g}.",
            "",
            "Uncertainty and prespecified references are included in published-report.json. No final-based reselection is permitted.",
        ]
        _figures(research / "final/predictions.parquet", final["serving_seed"], output)
    else:
        lines += [
            "",
            "## Final evaluation",
            "",
            "Pending. No final performance claim is available.",
        ]
    lines += [
        "",
        "## Limitations",
        "",
        "KolektorSDD2 uses controlled production images; physical-item/session identities are unverified. Scores are uncalibrated. Synthetic corruption and benchmark success do not establish factory deployment readiness.",
        "",
        "Dataset: https://www.vicos.si/resources/kolektorsdd2/ (CC BY-NC-SA 4.0).",
        "",
    ]
    (output / "study_report.md").write_text("\n".join(lines), encoding="utf-8")
    return {"report": str(output / "study_report.md"), "final_evaluated": final_path.exists()}


def _figures(path, seed, output):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from sklearn.metrics import precision_recall_curve

    from secom_quality.evaluation import capacity_metrics

    frame = pd.read_parquet(path)
    y, scores = frame.label.to_numpy(), frame[f"score_{seed}"].to_numpy()
    precision, recall, _ = precision_recall_curve(y, scores)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    axes[0].plot(recall, precision, label="Frozen serving model")
    axes[0].axhline(y.mean(), color="gray", linestyle="--", label="Prevalence")
    axes[0].set(xlabel="Recall", ylabel="Precision", title="Final precision-recall")
    fractions = np.linspace(0.01, 0.5, 50)
    values = [
        capacity_metrics(y, scores, frame.record_id.tolist(), float(f))["recall"] for f in fractions
    ]
    axes[1].plot(fractions, values, label="Frozen serving model")
    axes[1].plot(fractions, fractions, linestyle="--", color="gray", label="Expected random")
    axes[1].set(
        xlabel="Inspection capacity", ylabel="Failure recall", title="Final inspection capacity"
    )
    for axis in axes:
        axis.legend()
    figure.savefig(output / "final_curves.png", dpi=160)
    plt.close(figure)
