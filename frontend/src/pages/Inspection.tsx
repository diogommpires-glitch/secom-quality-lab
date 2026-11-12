import { useEffect, useRef, useState } from 'react';
import {
  AlertCircle,
  ArrowDownToLine,
  Check,
  ChevronRight,
  FileSpreadsheet,
  UploadCloud,
  X,
} from 'lucide-react';
import type { Batch, Prediction, ValidationIssue } from '../api';
import { API, ApiError, request } from '../api';
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
  percent,
} from '../components/ui';

interface BatchSummary {
  batch_id: string;
  row_count: number;
  created_at?: string;
}
export default function Inspection() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string[][]>([]);
  const [batch, setBatch] = useState<Batch | null>(null);
  const [issues, setIssues] = useState<ValidationIssue[]>([]);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [summaries, setSummaries] = useState<BatchSummary[]>([]);
  const [filter, setFilter] = useState('review');
  const [selected, setSelected] = useState<Prediction | null>(null);
  const [page, setPage] = useState(0);
  const requestId = useRef(crypto.randomUUID());
  const fileInput = useRef<HTMLInputElement>(null);
  useEffect(() => {
    void request<{ batches: BatchSummary[] }>('/batches')
      .then((v) => setSummaries(v.batches))
      .catch((e) => setError(e instanceof Error ? e.message : 'Could not load batches.'));
  }, []);
  async function chooseFile(next: File | null) {
    setFile(next);
    setError('');
    setIssues([]);
    setPreview([]);
    requestId.current = crypto.randomUUID();
    if (next) {
      try {
        const text = await next.text();
        setPreview(
          text
            .trim()
            .split(/\r?\n/)
            .slice(0, 4)
            .map((line) => line.split(',').slice(0, 7)),
        );
      } catch {
        setError('The selected file could not be read.');
      }
    }
  }
  async function submit() {
    if (!file || busy) return;
    setBusy(true);
    setError('');
    setIssues([]);
    const form = new FormData();
    form.set('file', file);
    form.set('client_request_id', requestId.current);
    try {
      const value = await request<Batch>('/batches', { method: 'POST', body: form });
      setBatch(value);
      setPage(0);
      setFilter('review');
      setSelected(null);
      const index = await request<{ batches: BatchSummary[] }>('/batches');
      setSummaries(index.batches);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Upload failed.');
      if (e instanceof ApiError) setIssues(e.issues);
    } finally {
      setBusy(false);
    }
  }
  async function openBatch(id: string) {
    setBusy(true);
    setError('');
    try {
      setBatch(await request<Batch>(`/batches/${encodeURIComponent(id)}`));
      setPage(0);
      setSelected(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not load the batch.');
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
    .sort((a, b) => (a.rank ?? Number.MAX_SAFE_INTEGER) - (b.rank ?? Number.MAX_SAFE_INTEGER));
  return (
    <>
      <div className="page-heading">
        <div>
          <span className="eyebrow">DEVELOPMENT INPUTS ONLY</span>
          <h1>Inspection queue</h1>
          <p>Score a batch and prioritise a fixed review capacity.</p>
        </div>
        <Badge tone="warning">Scores are not probabilities</Badge>
      </div>
      <Note>
        A queue recommendation is not a manufacturing pass or release decision. CSV validation is
        atomic: invalid input produces no scored batch.
      </Note>
      <Panel
        title="Import a batch"
        subtitle="Use the approved development sample or the exact sensor schema"
        action={
          <div className="actions">
            <LinkButton href={API + '/demo/inputs'}>Development sample</LinkButton>
            <LinkButton href={API + '/feature-schema/template'}>CSV template</LinkButton>
          </div>
        }
      >
        <div
          className="upload-area"
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault();
            void chooseFile(e.dataTransfer.files[0] ?? null);
          }}
        >
          <UploadCloud size={28} />
          <div>
            <h3>{file ? file.name : 'Bring your records into focus'}</h3>
            <p>
              {file
                ? `${format(file.size / 1024, 1)} KB · preview only until validation`
                : 'CSV with record_id and 590 named sensor columns. Missing sensor values are permitted.'}
            </p>
          </div>
          <input
            ref={fileInput}
            type="file"
            accept=".csv,text/csv"
            aria-label="Choose CSV file"
            className="file-input"
            onChange={(e) => void chooseFile(e.target.files?.[0] ?? null)}
          />
          <button
            className="button button-secondary"
            onClick={() => fileInput.current?.click()}
            disabled={busy}
          >
            {file ? 'Change file' : 'Choose CSV'}
          </button>
        </div>
        {preview.length > 0 && (
          <div className="preview">
            <span className="small-label">
              First rows · first 7 columns · server validates the complete file
            </span>
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    {preview[0].map((v, i) => (
                      <th key={i}>{v}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {preview.slice(1).map((row, i) => (
                    <tr key={i}>
                      {row.map((v, j) => (
                        <td key={j}>{v || 'missing'}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="upload-footer">
              <span>
                <FileSpreadsheet size={15} /> No labels are accepted as prediction inputs.
              </span>
              <button className="button" disabled={!file || busy} onClick={() => void submit()}>
                {busy ? 'Validating…' : 'Validate and score'}
                <ChevronRight size={16} />
              </button>
            </div>
          </div>
        )}
        {error && (
          <div className="inline-error" role="alert">
            <AlertCircle size={18} />
            <div>
              <strong>{error}</strong>
              {issues.length > 0 && (
                <ul>
                  {issues.slice(0, 20).map((issue, i) => (
                    <li key={i}>
                      Row {issue.row_index ?? issue.row ?? '—'}
                      {issue.field || issue.column ? ` · ${issue.field ?? issue.column}` : ''}:{' '}
                      {issue.message || issue.reason_code || issue.code}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </div>
        )}
      </Panel>
      {summaries.length > 0 && (
        <label className="batch-picker">
          Saved batch{' '}
          <select
            aria-label="Open saved batch"
            value={batch?.batch_id ?? ''}
            onChange={(e) => void openBatch(e.target.value)}
            disabled={busy}
          >
            <option value="" disabled>
              Select a batch…
            </option>
            {summaries.map((v) => (
              <option key={v.batch_id} value={v.batch_id}>
                {v.batch_id} · {v.row_count} records
              </option>
            ))}
          </select>
        </label>
      )}
      {busy && !file && <Loading label="Loading batch…" />}
      {batch ? (
        <>
          <div className="stats-grid">
            <Stat
              label="Batch records"
              value={format(batch.row_count)}
              note={batch.replayed ? 'Restored idempotent response' : 'Validated and recorded'}
            />
            <Stat
              label="Eligible records"
              value={format(batch.eligible_count)}
              note="Rows eligible for ranking"
            />
            <Stat
              label="Selected for review"
              value={format(batch.review_capacity)}
              note="Top 10% · floor of eligible count"
              accent
            />
            <Stat
              label="Abstained"
              value={format(batch.abstain_count)}
              note="Not ranked; inspect data quality"
            />
          </div>
          <Panel
            title="Prioritised records"
            subtitle={`Bundle ${batch.bundle_id}`}
            action={
              <a
                className="button button-secondary"
                href={`${API}/batches/${encodeURIComponent(batch.batch_id)}/export`}
              >
                <ArrowDownToLine size={15} /> Export queue
              </a>
            }
          >
            <div className="table-toolbar">
              <div className="segments" aria-label="Queue filter">
                {[
                  ['review', 'Review queue'],
                  ['all', 'All records'],
                  ['abstain', 'Abstained'],
                ].map(([value, label]) => (
                  <button
                    key={value}
                    className={filter === value ? 'active' : ''}
                    onClick={() => {
                      setFilter(value);
                      setPage(0);
                    }}
                  >
                    {label}
                  </button>
                ))}
              </div>
              <Badge>Fixed policy · 10%</Badge>
            </div>
            {rows.length > 0 ? (
              <div className="table-scroll">
                <table>
                  <thead>
                    <tr>
                      <th>Rank</th>
                      <th>Record</th>
                      <th>Risk score</th>
                      <th>Missingness</th>
                      <th>Data quality</th>
                      <th>Queue action</th>
                      <th />
                    </tr>
                  </thead>
                  <tbody>
                    {rows.slice(page * 15, page * 15 + 15).map((row) => (
                      <tr key={row.prediction_id}>
                        <td>{row.rank ?? '—'}</td>
                        <td>
                          <span className="table-strong">{row.record_id}</span>
                        </td>
                        <td>
                          <span className="score-number">{format(row.risk_score, 4)}</span>
                        </td>
                        <td>{percent(row.missingness)}</td>
                        <td>
                          <Badge tone={row.quality_status === 'abstain' ? 'warning' : 'neutral'}>
                            {row.quality_status}
                          </Badge>
                        </td>
                        <td>
                          {row.review_recommended ? (
                            <span className="review-action">
                              <Check size={14} /> Review
                            </span>
                          ) : (
                            <span className="muted">Not selected</span>
                          )}
                        </td>
                        <td>
                          <button
                            className="icon-button"
                            aria-label={`Inspect ${row.record_id}`}
                            onClick={() => setSelected(row)}
                          >
                            <ChevronRight size={16} />
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <Empty
                title={filter === 'abstain' ? 'No abstained records' : 'No records in this queue'}
              >
                Try a different queue filter. A small batch may have zero review capacity.
              </Empty>
            )}
            <div className="table-footer">
              <span>
                {format(rows.length)} records · {batch.batch_id}
              </span>
              <div>
                <button
                  className="button button-small button-secondary"
                  disabled={page === 0}
                  onClick={() => setPage((p) => p - 1)}
                >
                  Previous
                </button>
                <button
                  className="button button-small button-secondary"
                  disabled={(page + 1) * 15 >= rows.length}
                  onClick={() => setPage((p) => p + 1)}
                >
                  Next
                </button>
              </div>
            </div>
          </Panel>
        </>
      ) : (
        <Panel title="Your review queue" subtitle="A ranked batch will appear after validation">
          <Empty title="No batch selected">
            Download the approved development sample above to explore the complete workflow.
          </Empty>
        </Panel>
      )}
      {selected && (
        <div className="drawer-backdrop" onClick={() => setSelected(null)}>
          <section
            className="drawer"
            role="dialog"
            aria-modal="true"
            aria-labelledby="record-title"
            onClick={(e) => e.stopPropagation()}
            onKeyDown={(e) => {
              if (e.key === 'Escape') setSelected(null);
            }}
          >
            <div className="drawer-heading">
              <div>
                <span className="eyebrow">PREDICTION TRACE</span>
                <h2 id="record-title">{selected.record_id}</h2>
              </div>
              <button
                className="icon-button"
                autoFocus
                aria-label="Close record details"
                onClick={() => setSelected(null)}
              >
                <X size={20} />
              </button>
            </div>
            <MetricTable
              rows={[
                { label: 'Uncalibrated risk score', value: format(selected.risk_score, 5) },
                { label: 'Queue rank', value: selected.rank ?? 'Not ranked' },
                { label: 'Quality status', value: selected.quality_status },
                { label: 'Missingness', value: percent(selected.missingness) },
                { label: 'Range flags', value: format(selected.range_violation_count) },
                { label: 'Model', value: selected.model_id },
                { label: 'Bundle', value: <code>{selected.bundle_id}</code> },
                { label: 'Input fingerprint', value: <code>{selected.input_fingerprint}</code> },
              ]}
            />
            <Note>
              {selected.quality_reason_codes.length
                ? selected.quality_reason_codes.join(', ')
                : 'No quality reason codes were recorded.'}
            </Note>
          </section>
        </div>
      )}
    </>
  );
}
