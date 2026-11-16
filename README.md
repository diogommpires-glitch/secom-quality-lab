# SECOM Quality Lab

Code-only source snapshot of a semiconductor inspection prioritisation research application, with a Python service, React interface and industrial vision extension.

This repository contains source code, tests, dependency locks and third-party notices. Datasets, trained models, experiment outputs, historical reports, exports, local credentials and machine-specific launch configuration are excluded. Risk scores are uncalibrated. This is retrospective research and is not validated factory software.

The source includes the November 2026 audit corrections for metric boundaries, frozen-model provenance, JSON publication, release packaging, process ownership and application responsiveness. Ordinary imports are portable; AD2 can explicitly opt into a separately reviewed storage adapter. See [the artifact and metric contracts](docs/vision_artifact_contracts.md). Existing scientific outputs remain outside this code-only repository and were not replaced by the corrections.

## Local development

Use Python 3.12 and uv:

```powershell
uv sync --locked
uv run --locked pytest tests/test_science.py tests/test_common.py tests/test_application_inference.py tests/test_application_api.py tests/test_application_monitoring.py tests/test_store_version.py
```

For the interface, use Node.js and pnpm in `frontend`:

```powershell
pnpm install --frozen-lockfile
pnpm test
pnpm build
```

The frontend build writes the service's packaged web assets. No generated web bundle is included in this snapshot. Tests requiring retained datasets, model weights, GPU environments, experiment plans or historical evidence are outside this code-only snapshot. No research result is reproduced by publishing the code.

To build distributions with the viewer, run `pnpm build` in `frontend` before running `uv build` from the repository root. Both the source distribution and wheel then include the generated console assets. Trained models and datasets remain excluded.

GitHub Actions runs the explicit CPU/synthetic regression suite on Windows and Ubuntu, including the audit boundary regressions, plus Ruff formatting, lint and mypy. Optional Torch cases are skipped when that separate environment is absent. The frontend checks run on Ubuntu with Node.js 26.2.0 and pnpm 11.25.0. Action revisions and package-manager versions are pinned; dependency installation uses the committed lockfiles. Gitleaks 8.30.1 scans the checked Git tree with redacted findings. No dataset, GPU or paid-resource workflow is included.

Original project code uses the PolyForm Noncommercial License 1.0.0. See `LICENSE` and `LICENSING.md`. Dataset and third-party licences remain separate; see `THIRD_PARTY_NOTICES.md` and the vendored licence.
