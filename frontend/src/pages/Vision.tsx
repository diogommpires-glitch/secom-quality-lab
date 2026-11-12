import { useEffect, useRef, useState } from 'react';
import {
  Badge,
  Empty,
  LinkButton,
  Loading,
  MetricTable,
  Note,
  Panel,
  Stat,
  format,
} from '../components/ui';

const API = '/api/v2/vision';
interface VisionPrediction {
  prediction_id: string;
  record_id: string;
  risk_score: number | null;
  rank: number | null;
  review_recommended: boolean;
  quality_status: string;
  quality_reason_codes: string[];
  image_url: string;
  overlay_url: string | null;
  bundle_id: string;
  width: number;
  height: number;
}
interface VisionBatch {
  batch_id: string;
  row_count: number;
  eligible_count: number;
  abstain_count: number;
  review_capacity: number;
  bundle_id: string;
  predictions: VisionPrediction[];
}
interface VisionStatus {
  ready: boolean;
  message?: string;
  manifest?: Record<string, unknown>;
  report?: Record<string, unknown>;
}
interface ResearchProgress {
  state?: string;
  phase?: string;
  process_alive?: boolean | null;
  completed_fits?: number;
  failed_fits?: number;
  final_evaluated?: boolean;
  active_fit?: { config_id: string; seed: number; fold: number };
  training?: { config_id?: string; epoch?: number; epochs?: number };
}
async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(API + path, init);
  const body = await response.json();
  if (!response.ok)
    throw new Error(body.detail?.message ?? body.message ?? `Request failed (${response.status}).`);
  return body as T;
}
export default function Vision() {
  const [status, setStatus] = useState<VisionStatus | null>(null);
  const [progress, setProgress] = useState<ResearchProgress | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [capacity, setCapacity] = useState('0.1');
  const [batch, setBatch] = useState<VisionBatch | null>(null);
  const [selected, setSelected] = useState<VisionPrediction | null>(null);
  const [history, setHistory] = useState<{ batch_id: string; row_count: number }[]>([]);
  const [filter, setFilter] = useState('review');
  const [page, setPage] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const requestId = useRef(crypto.randomUUID());
  const ready = useRef(false);
  useEffect(() => {
    const updateStatus = (value: VisionStatus) => {
      ready.current = value.ready;
      setStatus(value);
    };
    const refresh = () =>
      void call<ResearchProgress>('/progress')
        .then((value) => {
          setProgress(value);
          if (value.final_evaluated && !ready.current)
            void fetch(API + '/status')
              .then((r) => r.json())
              .then(updateStatus)
              .catch(() => undefined);
        })
        .catch(() => undefined);
    refresh();
    const timer = window.setInterval(refresh, 15000);
    void fetch(API + '/status')
      .then((r) => r.json())
      .then(updateStatus)
      .catch(() => setError('The visual inspection service could not be reached.'));
    void call<{ batches: { batch_id: string; row_count: number }[] }>('/batches')
      .then((value) => setHistory(value.batches))
      .catch(() => undefined);
    return () => window.clearInterval(timer);
  }, []);
  async function submit() {
    if (!file || busy) return;
    setBusy(true);
    setError('');
    const form = new FormData();
    form.set('file', file);
    form.set('capacity', capacity);
    form.set('client_request_id', requestId.current);
    try {
      const value = await call<VisionBatch>('/batches', { method: 'POST', body: form });
      setBatch(value);
      setSelected(null);
      setPage(0);
      setHistory(
        (await call<{ batches: { batch_id: string; row_count: number }[] }>('/batches')).batches,
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Image scoring failed.');
    } finally {
      setBusy(false);
    }
  }
  const rows = (batch?.predictions ?? [])
    .filter(
      (row) =>
        filter === 'all' ||
        (filter === 'review' ? row.review_recommended : row.quality_status === 'abstain'),
    )
    .sort((a, b) => (a.rank ?? Infinity) - (b.rank ?? Infinity));
  return (
    <>
      <div className="page-heading">
        <div>
          <span className="eyebrow">INDUSTRIAL VISION RESEARCH</span>
          <h1>Visual inspection</h1>
          <p>Prioritise photographed production items for human review.</p>
        </div>
        <Badge tone={status?.ready ? 'good' : 'warning'}>
          {status?.ready ? 'Verified vision bundle' : 'No released vision model'}
        </Badge>
      </div>
      <Note>
        Visual inspection is a separate research task from SECOM sensor prediction. Scores are
        uncalibrated; localisation maps show model associations, not confidence or physical causes.
      </Note>
      {error && (
        <div className="state error-state" role="alert">
          {error}
        </div>
      )}
      {!status && !error && <Loading label="Checking vision bundle…" />}
      {status && !status.ready && (
        <Empty
          title={
            progress?.state === 'running'
              ? 'Training and evaluation in progress'
              : 'Vision research is not released'
          }
        >
          {progress?.state === 'running'
            ? 'Follow the current experiment below. Image scoring becomes available after evaluation.'
            : (status.message ??
              'A verified trained bundle is required before images can be scored.')}
        </Empty>
      )}
      {progress?.phase && (
        <Panel
          title="Research run"
          subtitle="Development and final evaluation have separate access boundaries"
        >
          <MetricTable
            rows={[
              { label: 'Phase', value: progress.phase.replaceAll('_', ' ') },
              {
                label: 'Run state',
                value:
                  progress.process_alive === false && progress.state === 'running'
                    ? 'Interrupted — resume the research command'
                    : (progress.state ?? 'Unknown'),
              },
              { label: 'Completed fits', value: String(progress.completed_fits ?? 0) },
              { label: 'Failed fits retained', value: String(progress.failed_fits ?? 0) },
              ...(progress.active_fit
                ? [
                    {
                      label: 'Current fit',
                      value: `${progress.active_fit.config_id}, fold ${progress.active_fit.fold + 1}, seed ${progress.active_fit.seed}`,
                    },
                  ]
                : []),
              ...(progress.training
                ? [
                    {
                      label: progress.training.config_id ?? 'Current model',
                      value: `Epoch ${progress.training.epoch ?? 0} of ${progress.training.epochs ?? 100}`,
                    },
                  ]
                : []),
            ]}
          />
          <p className="vision-progress-note">
            Development results guide selection. Final performance will appear when evaluation
            finishes.
          </p>
        </Panel>
      )}
      {status?.ready && (
        <>
          <Panel title="Active model and provenance" subtitle="Immutable bundle metadata">
            <MetricTable
              rows={['bundle_id', 'model_id', 'family', 'protocol_hash', 'dataset_hash']
                .filter((key) => status.manifest?.[key] !== undefined)
                .map((key) => ({
                  label: key.replaceAll('_', ' '),
                  value: String(status.manifest?.[key]),
                }))}
            />
            {status.manifest?.provenance !== undefined && (
              <pre className="vision-report">
                {JSON.stringify(status.manifest.provenance, null, 2)}
              </pre>
            )}
            <details>
              <summary>Published vision research report</summary>
              <pre className="vision-report">{JSON.stringify(status.report ?? {}, null, 2)}</pre>
            </details>
          </Panel>
          <Panel
            title="Import images"
            subtitle="PNG/JPEG or a flat ZIP; at most 100 images and 20 MiB"
          >
            <div className="vision-upload">
              <label>
                Image batch
                <input
                  aria-label="Image batch"
                  type="file"
                  accept=".zip,.png,.jpg,.jpeg"
                  disabled={busy}
                  onChange={(e) => {
                    setFile(e.target.files?.[0] ?? null);
                    requestId.current = crypto.randomUUID();
                    setError('');
                  }}
                />
              </label>
              <label>
                Inspection capacity
                <select
                  aria-label="Inspection capacity"
                  value={capacity}
                  disabled={busy}
                  onChange={(e) => {
                    setCapacity(e.target.value);
                    requestId.current = crypto.randomUUID();
                  }}
                >
                  <option value="0.1">10% of eligible images</option>
                  <option value="0.2">20% of eligible images</option>
                </select>
              </label>
              <button className="button" disabled={!file || busy} onClick={() => void submit()}>
                {busy ? 'Scoring images…' : 'Validate and score'}
              </button>
            </div>
            <p className="muted">
              Names identify records. Validation is atomic. Tiny batches may select zero images;
              abstentions have a separate workload.
            </p>
          </Panel>
        </>
      )}
      {history.length > 0 && (
        <Panel title="Saved image batches">
          <div className="actions">
            {history.slice(0, 8).map((item) => (
              <button
                className="button button-secondary"
                key={item.batch_id}
                onClick={() => {
                  setError('');
                  void call<VisionBatch>(`/batches/${item.batch_id}`)
                    .then((value) => {
                      setBatch(value);
                      setSelected(null);
                      setPage(0);
                    })
                    .catch((e: Error) => setError(e.message));
                }}
              >
                {item.batch_id.slice(0, 8)} · {item.row_count} images
              </button>
            ))}
          </div>
        </Panel>
      )}
      {batch && (
        <>
          <div className="stats-grid">
            <Stat
              label="Images"
              value={batch.row_count}
              note={`${batch.eligible_count} eligible for ranking`}
            />
            <Stat
              label="Inspect"
              value={batch.review_capacity}
              note="Exact capacity across eligible images"
              accent
            />
            <Stat
              label="Data-quality workload"
              value={batch.abstain_count}
              note="Excluded from ranking"
              warning={batch.abstain_count > 0}
            />
          </div>
          <Panel
            title="Inspection queue"
            subtitle={`Bundle ${batch.bundle_id}`}
            action={
              <LinkButton href={`${API}/batches/${batch.batch_id}/export`}>Export queue</LinkButton>
            }
          >
            <div className="actions">
              <label>
                Show{' '}
                <select
                  aria-label="Queue filter"
                  value={filter}
                  onChange={(e) => {
                    setFilter(e.target.value);
                    setPage(0);
                  }}
                >
                  <option value="review">Selected for inspection</option>
                  <option value="all">All images</option>
                  <option value="abstain">Data-quality workload</option>
                </select>
              </label>
            </div>
            {rows.length === 0 ? (
              <Empty title="No images in this view">
                A small batch may have zero inspection capacity. Choose all images to inspect
                individual results.
              </Empty>
            ) : (
              <div className="table-scroll">
                <table>
                  <thead>
                    <tr>
                      <th>Image</th>
                      <th>Record</th>
                      <th>Rank</th>
                      <th>Risk score</th>
                      <th>Quality</th>
                      <th>Review</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.slice(page * 25, page * 25 + 25).map((row) => (
                      <tr key={row.prediction_id}>
                        <td>
                          <img
                            className="vision-thumb"
                            src={row.image_url}
                            alt={`Record ${row.record_id}`}
                            loading="lazy"
                          />
                        </td>
                        <td>
                          <button className="text-button" onClick={() => setSelected(row)}>
                            {row.record_id}
                          </button>
                        </td>
                        <td>{row.rank ?? '—'}</td>
                        <td>{format(row.risk_score, 4)}</td>
                        <td>{row.quality_status}</td>
                        <td>{row.review_recommended ? 'Inspect' : 'Not selected'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            {rows.length > 25 && (
              <div className="actions">
                <button
                  className="button button-secondary"
                  disabled={page === 0}
                  onClick={() => setPage(page - 1)}
                >
                  Previous
                </button>
                <span>
                  Page {page + 1} of {Math.ceil(rows.length / 25)}
                </span>
                <button
                  className="button button-secondary"
                  disabled={(page + 1) * 25 >= rows.length}
                  onClick={() => setPage(page + 1)}
                >
                  Next
                </button>
              </div>
            )}
          </Panel>
          {selected && (
            <Panel
              title={`Record ${selected.record_id}`}
              subtitle={`${selected.width} × ${selected.height} pixels`}
            >
              <div className="vision-images">
                <figure>
                  <img src={selected.image_url} alt={`Original ${selected.record_id}`} />
                  <figcaption>Original image</figcaption>
                </figure>
                {selected.overlay_url && (
                  <figure>
                    <img
                      src={selected.overlay_url}
                      alt={`Model localisation for ${selected.record_id}`}
                    />
                    <figcaption>Model localisation · display scale normalised per image</figcaption>
                  </figure>
                )}
              </div>
              <MetricTable
                rows={[
                  { label: 'Risk score', value: format(selected.risk_score, 4) },
                  { label: 'Quality', value: selected.quality_status },
                  { label: 'Reasons', value: selected.quality_reason_codes.join(', ') || 'None' },
                  { label: 'Bundle', value: selected.bundle_id },
                ]}
              />
            </Panel>
          )}
        </>
      )}
    </>
  );
}
