#!/usr/bin/env bash
# Start only after the live self-stop/volume-retention test succeeds.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STUDY_ROOT="${SECOM_WORKSPACE:-/workspace/secom-cloud-study}"
export SECOM_GPU_IDS="${SECOM_GPU_IDS:-0}"
export TORCH_HOME="${TORCH_HOME:-/workspace/secom-torch-cache}"
export TMPDIR="${TMPDIR:-/workspace/secom-tmp}"
mkdir -p "$TORCH_HOME" "$TMPDIR"
: "${RUNPOD_POD_ID:?RunPod must supply its own Pod ID}"
: "${SECOM_HOURLY_USD:?Use the verified total Pod hourly price}"
: "${SECOM_BILLING_STARTED_UNIX:?Use the paid Pod creation timestamp}"
mkdir -p "$STUDY_ROOT/vision/runtime/runpod"
cd "$PROJECT_ROOT"
python3 - "$PROJECT_ROOT" "$RUNPOD_POD_ID" <<'PY'
import json,sys,os,math,time
from pathlib import Path
record=Path(sys.argv[1])/'watchdog-live-verification.json'
if not record.exists(): raise SystemExit('First perform the live Pod self-stop, retention and resume test.')
data=json.loads(record.read_text())
if data.get('pod_id')!=sys.argv[2] or not data.get('provider_stop_verified') or not data.get('volume_retention_verified'):
    raise SystemExit('Watchdog verification does not cover this Pod and persistent volume.')
rate=float(os.environ['SECOM_HOURLY_USD']); started=float(os.environ['SECOM_BILLING_STARTED_UNIX'])
budget=float(os.environ.get('SECOM_BUDGET_USD','100')); hours=float(os.environ.get('SECOM_MAX_HOURS','40'))
if not all(math.isfinite(x) for x in (rate,started,budget,hours)) or not (0<rate<=2.5 and 0<budget<=100 and 0<hours<=40 and started<=time.time()+60):
    raise SystemExit('Invalid price/start-time/budget: explicitly review the initial$100/40hour boundary.')
PY
if [[ -f "$STUDY_ROOT/vision/runtime/STOP_REQUESTED" ]]; then
  echo 'Explicitly review the stop reason and budget before removing STOP_REQUESTED to resume.' >&2
  exit 2
fi
if [[ "${1:-}" != '--guard-only' && -f "$STUDY_ROOT/vision/runtime/runpod/job.pid" ]] && kill -0 "$(cat "$STUDY_ROOT/vision/runtime/runpod/job.pid")" 2>/dev/null; then
  echo 'A research process is already running; refusing a duplicate launch.' >&2
  exit 2
fi
verify_guard() {
  python3 scripts/runpod/watchdog.py --verify --workspace "$STUDY_ROOT" \
    --pod-id "$RUNPOD_POD_ID" --started-unix "$SECOM_BILLING_STARTED_UNIX" \
    --hourly-usd "$SECOM_HOURLY_USD" --budget-usd "${SECOM_BUDGET_USD:-100}" \
    --max-hours "${SECOM_MAX_HOURS:-40}"
}
if [[ ! -f "$STUDY_ROOT/vision/runtime/runpod/watchdog.pid" ]] || ! kill -0 "$(cat "$STUDY_ROOT/vision/runtime/runpod/watchdog.pid")" 2>/dev/null; then
nohup python3 -u scripts/runpod/watchdog.py --workspace "$STUDY_ROOT" \
  --pod-id "$RUNPOD_POD_ID" --started-unix "$SECOM_BILLING_STARTED_UNIX" \
  --hourly-usd "$SECOM_HOURLY_USD" --budget-usd "${SECOM_BUDGET_USD:-100}" \
  --max-hours "${SECOM_MAX_HOURS:-40}" \
  > "$STUDY_ROOT/vision/runtime/runpod/watchdog.log" 2>&1 < /dev/null &
echo "$!" > "$STUDY_ROOT/vision/runtime/runpod/watchdog.pid"
sleep 1
fi
verify_guard
if [[ "${1:-}" == '--guard-only' ]]; then
  echo 'Pod-side guard is active for installation, download, idle and training time.'
  exit 0
fi
python3 scripts/runpod/dev_cache.py recover --workspace "$STUDY_ROOT" > /dev/null
"$PROJECT_ROOT/.venv-runpod/bin/python" - <<'PY'
import json, os
from scripts.runpod.pilot import check_hardware
print(json.dumps(check_hardware(os.environ['SECOM_GPU_IDS']), indent=2))
PY
ARGS=()
if [[ "${SECOM_SKIP_AD2:-0}" == '1' ]]; then
  ARGS+=(--skip-ad2)
elif [[ ! -f "$STUDY_ROOT/vision/ad2/source/manifest.json" ]]; then
  if ! "$PROJECT_ROOT/.venv-runpod/bin/python" - "$STUDY_ROOT" <<'AD2_PREPARATION_CHECK'
import json, os, sys
from pathlib import Path
if os.environ.get('SECOM_AD2_BACKGROUND_PREPARATION') != '1':
    raise SystemExit('Acquire and audit AD2 first, or explicitly choose SECOM_SKIP_AD2=1.')
marker = Path(sys.argv[1])/'vision/runtime/runpod/ad2-preparation.json'
try:
    preparation = json.loads(marker.read_text(encoding='utf-8'))
except (OSError, ValueError):
    raise SystemExit('Background AD2 preparation needs a readable, valid preparation marker.') from None
expected_sha256 = 'c0ded99ef32bfc8e352d52beb44515e5b292b8598cb963aadfa91ca0763505e4'
if not isinstance(preparation, dict) or (
    preparation.get('state') != 'preparing'
    or preparation.get('archive_sha256') != expected_sha256
    or preparation.get('scope') != 'development_source_audit_only'
    or type(preparation.get('pid')) is not int
    or preparation['pid'] <= 0
):
    raise SystemExit('Background AD2 preparation marker must cover the pinned source-only audit and a positive PID.')
try:
    os.kill(preparation['pid'], 0)
except OSError:
    raise SystemExit('Background AD2 preparation process is not verifiably alive.') from None
print('Background AD2 source acquisition/audit is active; the AD2 challenge remains enabled.')
AD2_PREPARATION_CHECK
  then
    exit 2
  fi
fi
nohup setsid "$PROJECT_ROOT/.venv-runpod/bin/python" -u scripts/runpod/parallel_study.py \
  --workspace "$STUDY_ROOT" --gpus "$SECOM_GPU_IDS" "${ARGS[@]}" \
  > "$STUDY_ROOT/vision/runtime/runpod/research.log" 2>&1 < /dev/null &
MONITOR_RUNNING="$(python3 - "$STUDY_ROOT" <<'MONITOR_CHECK'
import sys
from pathlib import Path
try:
    marker = Path(sys.argv[1])/'vision/runtime/runpod/live-monitor.pid'
    pid = int(marker.read_text())
    command = (Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0')
    print('1' if any(Path(part.decode(errors='replace')).name == 'live_monitor.py' for part in command) else '0')
except (OSError, ValueError):
    print('0')
MONITOR_CHECK
)"
if [[ "$MONITOR_RUNNING" != '1' ]]; then
  nohup python3 -u scripts/runpod/live_monitor.py --workspace "$STUDY_ROOT" --interval 5 \
    > "$STUDY_ROOT/vision/runtime/runpod/live-monitor.log" 2>&1 < /dev/null &
  echo "$!" > "$STUDY_ROOT/vision/runtime/runpod/live-monitor.pid"
fi
echo 'Detached research, read-only telemetry and Pod-side budget guard started. No account-management key was uploaded.'
