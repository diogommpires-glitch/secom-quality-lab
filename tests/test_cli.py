import json

from typer.testing import CliRunner

from secom_quality.cli import app


def test_doctor_reports_runtime_without_creating_workspace(tmp_path):
    workspace = tmp_path / "absent"
    result = CliRunner().invoke(app, ["--workspace", str(workspace), "doctor"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["python"].startswith("3.12.")
    assert report["workspace"] == str(workspace.resolve())
    assert "uncalibrated" in report["limitation"]
    assert not workspace.exists()


def test_lifecycle_commands_are_exposed():
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in (
        "doctor",
        "data",
        "protocol",
        "experiment",
        "selection",
        "final",
        "report",
        "bundle",
        "predict",
        "replay",
        "serve",
    ):
        assert command in result.output
