import { useMemo, useState } from 'react';
import { Search, ShieldCheck } from 'lucide-react';
import type { Json, JsonObject } from '../api';
import { arrayAt, numberAt, objectAt, textAt } from '../api';
import Chart from '../components/Chart';
import { Badge, MetricTable, Note, Panel, Stat, format, percent } from '../components/ui';

function featureMissing(value: Json, index: number, n: number): { name: string; fraction: number } {
  if (typeof value === 'number')
    return {
      name: `sensor_${String(index).padStart(3, '0')}`,
      fraction: value > 1 ? value / n : value,
    };
  if (value && typeof value === 'object' && !Array.isArray(value))
    return {
      name: textAt(
        value,
        'feature',
        textAt(value, 'sensor', `sensor_${String(index).padStart(3, '0')}`),
      ),
      fraction:
        numberAt(value, 'missing_fraction') ??
        numberAt(value, 'missingness') ??
        (numberAt(value, 'missing_count') ?? 0) / n,
    };
  return { name: `sensor_${String(index).padStart(3, '0')}`, fraction: 0 };
}
export default function DataAudit({ audit }: { audit: JsonObject }) {
  const [query, setQuery] = useState('');
  const [page, setPage] = useState(0);
  const n = numberAt(audit, 'n_samples') ?? 1;
  const features = numberAt(audit, 'n_features') ?? 1;
  const partitions = objectAt(audit, 'partitions');
  const development = objectAt(partitions, 'development');
  const final = objectAt(partitions, 'final');
  const missing = useMemo(
    () =>
      arrayAt(audit, 'missingness_by_feature')
        .map((v, i) => featureMissing(v, i, n))
        .sort((a, b) => b.fraction - a.fraction),
    [audit, n],
  );
  const filtered = missing.filter((v) => v.name.includes(query.trim().toLowerCase()));
  const shown = filtered.slice(page * 12, page * 12 + 12);
  return (
    <>
      <div className="page-heading">
        <div>
          <span className="eyebrow">SOURCE INTEGRITY</span>
          <h1>Data audit</h1>
          <p>Inspect the source before interpreting the model.</p>
        </div>
        <Badge tone="good">
          <ShieldCheck size={14} /> Checksummed source
        </Badge>
      </div>
      <div className="stats-grid">
        <Stat
          label="Observations"
          value={format(n)}
          note="One supplied production entity per row"
        />
        <Stat label="Features" value={format(features)} note="Named by index, not sensor physics" />
        <Stat
          label="Missing cells"
          value={format(numberAt(audit, 'missing_cells'))}
          note={`${percent((numberAt(audit, 'missing_cells') ?? 0) / (n * features))} of feature cells`}
        />
        <Stat
          label="Timestamp tie groups"
          value={format(numberAt(audit, 'timestamp_tie_groups'))}
          note="Kept together at split boundaries"
        />
      </div>
      <Panel
        title="Chronological partitions"
        subtitle="Development first; the later partition is held outside model selection"
      >
        <div className="partition-bar">
          <div style={{ flex: numberAt(development, 'n_samples') ?? 1 }}>
            Development <strong>{format(numberAt(development, 'n_samples'))}</strong>
          </div>
          <div style={{ flex: numberAt(final, 'n_samples') ?? 1 }}>
            Final holdout <strong>{format(numberAt(final, 'n_samples'))}</strong>
          </div>
        </div>
        <div className="two-col partition-details">
          <MetricTable
            rows={[
              { label: 'Development failures', value: format(numberAt(development, 'n_failures')) },
              { label: 'Starts', value: textAt(development, 'start') },
              { label: 'Ends', value: textAt(development, 'end') },
            ]}
          />
          <MetricTable
            rows={[
              { label: 'Final failures', value: format(numberAt(final, 'n_failures')) },
              { label: 'Starts', value: textAt(final, 'start') },
              { label: 'Ends', value: textAt(final, 'end') },
            ]}
          />
        </div>
      </Panel>
      <div className="two-col">
        <Panel
          title="Missingness by sensor"
          subtitle="Development rows only · highest missing fractions"
        >
          {missing.length > 0 ? (
            <Chart
              label="Most missing sensor columns"
              data={[
                {
                  type: 'bar',
                  orientation: 'h',
                  y: missing
                    .slice(0, 10)
                    .reverse()
                    .map((v) => v.name),
                  x: missing
                    .slice(0, 10)
                    .reverse()
                    .map((v) => v.fraction),
                  marker: { color: '#4e8c7d' },
                  hovertemplate: '%{y}: %{x:.1%}<extra></extra>',
                },
              ]}
              layout={{
                margin: { l: 85, r: 20, t: 10, b: 35 },
                xaxis: { tickformat: '.0%', gridcolor: '#edf0f4' },
                yaxis: { gridcolor: 'transparent' },
              }}
            />
          ) : (
            <p className="panel-caption">Feature missingness has not been recorded.</p>
          )}
        </Panel>
        <Panel title="Source checks" subtitle="Data facts, not model-selection rules">
          <MetricTable
            rows={[
              { label: 'First timestamp', value: textAt(audit, 'timestamp_min') },
              { label: 'Last timestamp', value: textAt(audit, 'timestamp_max') },
              { label: 'Unique timestamps', value: format(numberAt(audit, 'unique_timestamps')) },
              {
                label: 'Largest timestamp tie',
                value: format(numberAt(audit, 'max_timestamp_tie')),
              },
              {
                label: 'Exact duplicate groups',
                value: format(numberAt(audit, 'exact_duplicate_groups')),
              },
            ]}
          />
          <Note>
            {textAt(
              audit,
              'metadata_discrepancy',
              'Source files determine the actual feature schema.',
            )}
          </Note>
        </Panel>
      </div>
      <Panel
        title="Sensor missingness register"
        subtitle="Development-only feature statistics"
        action={
          <label className="search">
            <Search size={15} />
            <input
              aria-label="Search sensors"
              placeholder="Find a sensor…"
              value={query}
              onChange={(e) => {
                setQuery(e.target.value);
                setPage(0);
              }}
            />
          </label>
        }
      >
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Feature identifier</th>
                <th>Missing fraction</th>
                <th>Coverage</th>
              </tr>
            </thead>
            <tbody>
              {shown.map((row) => (
                <tr key={row.name}>
                  <td>
                    <code>{row.name}</code>
                  </td>
                  <td>{percent(row.fraction)}</td>
                  <td>
                    <div className="coverage-bar">
                      <span style={{ width: `${(1 - row.fraction) * 100}%` }} />
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="table-footer">
          <span>
            {filtered.length
              ? `${page * 12 + 1}–${Math.min((page + 1) * 12, filtered.length)} of ${filtered.length} sensors`
              : 'No matching sensors'}
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
              disabled={(page + 1) * 12 >= filtered.length}
              onClick={() => setPage((p) => p + 1)}
            >
              Next
            </button>
          </div>
        </div>
      </Panel>
      <Panel title="Source fingerprints" subtitle="Download and input integrity">
        <div className="hash-list">
          {Object.entries(objectAt(audit, 'source_hashes')).map(([key, value]) => (
            <div key={key}>
              <span>{key}</span>
              <code>{String(value)}</code>
            </div>
          ))}
        </div>
      </Panel>
    </>
  );
}
