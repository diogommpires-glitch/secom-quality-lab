# SECOM Quality Lab

Code-only source snapshot of a semiconductor inspection prioritisation research application, with a Python service, React interface and industrial vision extension.

This repository contains source code, tests, dependency locks and third-party notices. Datasets, trained models, experiment outputs, historical reports, exports, local credentials and machine-specific launch configuration are excluded. Risk scores are uncalibrated. This is retrospective research and is not validated factory software.

The snapshot uses the current source tree except for `src/secom_quality/common.py` and `src/secom_quality/vision/ad2.py`, which come from source commit `113a3b3`. Their installed versions depend on a separate local dataset relocation binding. The original application has not been changed.

## Local development

Use Python 3.12 and uv:

```powershell
uv sync --locked
uv run --locked pytest tests/test_science.py tests/test_application_inference.py tests/test_application_api.py tests/test_application_monitoring.py tests/test_store_version.py
```

For the interface, use Node.js and pnpm in `frontend`:

```powershell
pnpm install --frozen-lockfile
pnpm test
pnpm build
```

The frontend build writes the service's packaged web assets. No generated web bundle is included in this snapshot. Tests requiring retained datasets, model weights, GPU environments, experiment plans or historical evidence are outside this code-only snapshot. No research result is reproduced by publishing the code.

GitHub Actions runs the selected 34 Python tests on Windows and Ubuntu, plus Ruff and mypy. The frontend checks run on Ubuntu with Node.js 26.2.0 and pnpm 11.25.0. Action revisions and package-manager versions are pinned; dependency installation uses the committed lockfiles. Gitleaks 8.30.1 scans the checked Git tree with redacted findings. No dataset, GPU or paid-resource workflow is included.

Original project code uses the PolyForm Noncommercial License 1.0.0. See `LICENSE` and `LICENSING.md`. Dataset and third-party licences remain separate; see `THIRD_PARTY_NOTICES.md` and the vendored licence.
