import { useEffect, useRef, useState } from 'react';
import { Activity, ChevronRight, Pause, Play, RotateCcw } from 'lucide-react';
import type { JsonObject, StreamState } from '../api';
import { arrayAt, numberAt, objectAt, post, request, textAt } from '../api';
import Chart from '../components/Chart';
import {
  Badge,
  Empty,
  ErrorState,
  MetricTable,
  Note,
  Panel,
  Stat,
  format,
  percent,
} from '../components/ui';

const scenarios = [
  {
    id: 'baseline',
    title: 'Baseline replay',
    description: 'Historical development observations with simulated delayed outcomes.',
  },
  {
    id: 'missingness',
    title: 'Missingness stress',
    description: 'Injected missing sensor values; explicitly synthetic corruption.',
  },
  {
    id: 'sensor_shift',
    title: 'Sensor shift stress',
    description: 'Injected measurement shift; no claim of a real process change.',
  },
];
function featureSignalLabel(feature: JsonObject): string {
  if (numberAt(feature, 'count') === null) return 'No window';
  if (feature.complete !== true) return 'Warming up';
  if (feature.warning === true) return 'Warning';
  return 'No alert';
}
export default function Monitoring() {
  const [scenario, setScenario] = useState('baseline');
  const [stream, setStream] = useState<StreamState | null>(null);
  const [streams, setStreams] = useState<StreamState[]>([]);
  const [windows, setWindows] = useState<JsonObject[]>([]);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [playing, setPlaying] = useState(false);
  const advancing = useRef(false);
  useEffect(() => {
    void request<{ streams: StreamState[] }>('/monitoring/streams')
      .then((v) => setStreams(v.streams))
      .catch((e) => setError(e instanceof Error ? e.message : 'Could not load replay sessions.'));
  }, []);
  async function loadWindows(id: string) {
    const value = await request<{ windows: JsonObject[] }>(
      `/monitoring/streams/${encodeURIComponent(id)}/windows`,
    );
    setWindows(value.windows);
  }
  async function create() {
    if (busy) return;
    setBusy(true);
    setPlaying(false);
    setError('');
    try {
      const value = await post<StreamState>('/demo/replays', { scenario });
      setStream(value);
      setWindows([]);
      setStreams((old) => [value, ...old.filter((v) => v.stream_id !== value.stream_id)]);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Replay could not be created.');
    } finally {
      setBusy(false);
    }
  }
  async function open(id: string) {
    setPlaying(false);
    setError('');
    setBusy(true);
    try {
      const value = await request<StreamState>(`/monitoring/streams/${encodeURIComponent(id)}`);
      setStream(value);
      setScenario(value.scenario);
      await loadWindows(id);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not load replay.');
    } finally {
      setBusy(false);
    }
  }
  async function advance() {
    if (!stream || advancing.current || stream.status === 'completed') return;
    advancing.current = true;
    setBusy(true);
    setError('');
    try {
      const value = await post<StreamState>(
        `/demo/replays/${encodeURIComponent(stream.stream_id)}/advance`,
        { expected_cursor: stream.cursor, steps: 20 },
      );
      setStream(value);
      await loadWindows(value.stream_id);
      if (value.status === 'completed') setPlaying(false);
    } catch (e) {
      setPlaying(false);
      setError(e instanceof Error ? e.message : 'Replay could not advance.');
    } finally {
      advancing.current = false;
      setBusy(false);
    }
  }
  async function togglePlayback() {
    if (!stream || busy) return;
    if (!playing) {
      setPlaying(true);
      return;
    }
    setPlaying(false);
    try {
      setStream(
        await post<StreamState>(`/demo/replays/${encodeURIComponent(stream.stream_id)}/pause`, {}),
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : 'The replay could not pause.');
    }
  }
  useEffect(() => {
    if (!playing || !stream || busy) return;
    const timer = window.setTimeout(() => void advance(), 900);
    return () => window.clearTimeout(timer);
  });
  const latest = stream ? objectAt(stream, 'latest_window') : {};
  const feature = objectAt(latest, 'feature_window');
  const performance = objectAt(latest, 'performance_window');
  const metrics = objectAt(performance, 'metrics');
  const status = textAt(performance, 'status', 'insufficient_data');
  const complete = !!stream && stream.status === 'completed';
  return (
    <>
      <div className="page-heading">
        <div>
          <span className="eyebrow">CONTROLLED REPLAY</span>
          <h1>Monitoring</h1>
          <p>Watch observations arrive, then outcomes mature.</p>
        </div>
        <Badge tone="warning">Demo stream · virtual time</Badge>
      </div>
      <Note>
        Replay uses development data and a separate demo model. Outcomes are delayed by two batches.
        Stress scenarios are synthetic; alerts do not establish real process or concept drift.
      </Note>
      <Panel
        title="Replay scenario"
        subtitle="Create a session, then advance records deterministically"
      >
        <div className="scenario-grid">
          {scenarios.map((item) => (
            <button
              key={item.id}
              className={`scenario ${scenario === item.id ? 'selected' : ''}`}
              onClick={() => setScenario(item.id)}
              aria-pressed={scenario === item.id}
              disabled={busy || playing}
            >
              <Activity size={19} />
              <strong>{item.title}</strong>
              <span>{item.description}</span>
            </button>
          ))}
        </div>
        <div className="replay-toolbar">
          <button className="button" onClick={() => void create()} disabled={busy || playing}>
            <RotateCcw size={15} /> New replay
          </button>
          {streams.length > 0 && (
            <label className="inline-label">
              Resume{' '}
              <select
                aria-label="Resume replay"
                value={stream?.stream_id ?? ''}
                onChange={(e) => void open(e.target.value)}
                disabled={busy || playing}
              >
                <option value="" disabled>
                  Select a session…
                </option>
                {streams.map((item) => (
                  <option key={item.stream_id} value={item.stream_id}>
                    {item.scenario} · {item.stream_id.slice(-10)}
                  </option>
                ))}
              </select>
            </label>
          )}
        </div>
      </Panel>
      {error && (
        <ErrorState
          message={error}
          retry={stream ? () => void open(stream.stream_id) : () => void create()}
        />
      )}
      {stream ? (
        <>
          <div className="replay-control">
            <div className="replay-title">
              <Badge tone={complete ? 'good' : 'neutral'}>
                {complete ? 'Complete' : playing ? 'Playing' : 'Paused'}
              </Badge>
              <strong>{stream.scenario.replaceAll('_', ' ')}</strong>
              <span>{stream.stream_id}</span>
            </div>
            <div className="actions">
              <button
                className="button button-secondary"
                disabled={busy || complete}
                onClick={() => void togglePlayback()}
              >
                {playing ? <Pause size={15} /> : <Play size={15} />} {playing ? 'Pause' : 'Play'}
              </button>
              <button
                className="button"
                disabled={busy || playing || complete}
                onClick={() => void advance()}
              >
                {stream.status === 'flushing' ? 'Flush outcomes' : 'Step 20'}{' '}
                <ChevronRight size={15} />
              </button>
            </div>
            <div className="progress-track">
              <span
                style={{
                  width: `${(Math.min(stream.cursor, stream.total_records) / Math.max(stream.total_records, 1)) * 100}%`,
                }}
              />
            </div>
            <span className="small-text">
              {format(Math.min(stream.cursor, stream.total_records))} /{' '}
              {format(stream.total_records)} observations · cursor checked on every advance
            </span>
          </div>
          <div className="stats-grid">
            <Stat
              label="Arrived observations"
              value={format(numberAt(latest, 'arrived_count') ?? stream.cursor)}
              note="Predictions are recorded first"
            />
            <Stat
              label="Matured outcomes"
              value={format(numberAt(latest, 'matured_label_count') ?? 0)}
              note="Two-batch simulated delay"
            />
            <Stat
              label="Pending outcomes"
              value={format(numberAt(latest, 'pending_label_count') ?? 0)}
              note="No outcome means no performance claim"
            />
            <Stat
              label="Feature-window signal"
              value={featureSignalLabel(feature)}
              note={
                numberAt(feature, 'count') == null
                  ? 'No feature window yet'
                  : `${format(numberAt(feature, 'count'))} observations in window`
              }
              accent={feature.complete === true && feature.warning !== true}
              warning={feature.complete === true && feature.warning === true}
            />
          </div>
          <div className="two-col">
            <Panel
              title="Feature health"
              subtitle="Recent input window · separate from outcome performance"
            >
              <MetricTable
                rows={[
                  { label: 'Window observations', value: format(numberAt(feature, 'count')) },
                  {
                    label: 'Current missingness',
                    value: percent(numberAt(feature, 'missingness')),
                  },
                  {
                    label: 'Reference missingness',
                    value: percent(numberAt(feature, 'reference_missingness')),
                  },
                  {
                    label: 'Missingness change',
                    value: percent(numberAt(feature, 'missingness_delta')),
                  },
                  { label: 'Score KS statistic', value: format(numberAt(feature, 'score_ks'), 3) },
                  {
                    label: 'Consecutive warning windows',
                    value: format(numberAt(feature, 'consecutive_warning_windows')),
                  },
                ]}
              />
              <Note>
                A distribution warning is a diagnostic signal, not evidence that model accuracy has
                changed.
              </Note>
            </Panel>
            <Panel
              title="Outcome performance"
              subtitle="Matured labels only"
              action={
                <Badge tone={status === 'available' ? 'good' : 'warning'}>
                  {status.replaceAll('_', ' ')}
                </Badge>
              }
            >
              <MetricTable
                rows={[
                  { label: 'Labelled observations', value: format(numberAt(performance, 'count')) },
                  {
                    label: 'Observed failures',
                    value: format(numberAt(performance, 'positive_count')),
                  },
                  {
                    label: 'Observed passes',
                    value: format(numberAt(performance, 'negative_count')),
                  },
                  { label: 'Outcome coverage', value: percent(numberAt(performance, 'coverage')) },
                  {
                    label: 'Sequence coverage',
                    value: `${format(numberAt(performance, 'first_sequence'))} – ${format(numberAt(performance, 'last_sequence'))}`,
                  },
                  {
                    label: 'Average precision',
                    value: format(numberAt(metrics, 'average_precision'), 3),
                  },
                ]}
              />
              {Object.keys(metrics).length === 0 && (
                <Note>
                  Insufficient matured outcomes for performance. Missing metrics are not zero error.
                </Note>
              )}
            </Panel>
          </div>
          <Panel title="Window history" subtitle="Measured replay diagnostics">
            {windows.length > 0 ? (
              <Chart
                label="Replay missingness across observed cursors"
                data={[
                  {
                    type: 'scatter',
                    mode: 'lines+markers',
                    x: windows.map((v) => numberAt(v, 'cursor') ?? 0),
                    y: windows.map(
                      (v) => numberAt(objectAt(v, 'feature_window'), 'missingness') ?? 0,
                    ),
                    name: 'Missingness',
                    line: { color: '#277464', width: 2 },
                    marker: { size: 4 },
                  },
                ]}
                layout={{
                  xaxis: { title: { text: 'Arrived observations' }, gridcolor: '#edf0f4' },
                  yaxis: {
                    title: { text: 'Feature missingness' },
                    tickformat: '.1%',
                    gridcolor: '#edf0f4',
                  },
                }}
              />
            ) : (
              <Empty title="No monitoring windows yet">
                Advance the replay to generate a window.
              </Empty>
            )}
          </Panel>
          {arrayAt(feature, 'feature_summaries').length > 0 && (
            <Panel
              title="Selected sensor summaries"
              subtitle="Anonymous identifiers; changes are not physical diagnoses"
            >
              <div className="table-scroll">
                <table>
                  <thead>
                    <tr>
                      <th>Sensor</th>
                      <th>Reference mean</th>
                      <th>Current mean</th>
                    </tr>
                  </thead>
                  <tbody>
                    {arrayAt(feature, 'feature_summaries').map((item, i) => {
                      if (!item || typeof item !== 'object' || Array.isArray(item)) return null;
                      return (
                        <tr key={i}>
                          <td>
                            <code>{textAt(item, 'sensor')}</code>
                          </td>
                          <td>{format(numberAt(item, 'reference_mean'), 4)}</td>
                          <td>{format(numberAt(item, 'current_mean'), 4)}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </Panel>
          )}
        </>
      ) : (
        <Panel title="Replay workspace" subtitle="No monitoring session selected">
          <Empty title="Start with a baseline">
            Create a replay above. Each step records predictions and releases only outcomes whose
            simulated delay has elapsed.
          </Empty>
        </Panel>
      )}
    </>
  );
}
