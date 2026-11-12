import {
  ArrowRight,
  CheckCircle2,
  FileText,
  FlaskConical,
  GitBranch,
  Layers3,
  LockKeyhole,
} from 'lucide-react';
import type { Json, JsonObject, ModelInfo, ExperimentData } from '../api';
import { arrayAt, numberAt, objectAt, textAt } from '../api';
import Chart from '../components/Chart';
import { Badge, Empty, MetricTable, Note, Panel, Stat, format, percent } from '../components/ui';
import type { Page } from '../App';

export interface ResearchData {
  model: ModelInfo;
  audit: JsonObject;
  experiments: ExperimentData;
  report: JsonObject;
}
export function Overview({
  data,
  navigate,
}: {
  data: ResearchData;
  navigate: (page: Page) => void;
}) {
  const { audit, experiments, model, report } = data;
  const classes = objectAt(audit, 'class_counts');
  const primary = objectAt(experiments.final_metrics ?? {}, 'primary');
  const finalAvailable =
    experiments.final_status !== 'pending' && numberAt(primary, 'n_samples') !== null;
  const randomReview = objectAt(
    objectAt(experiments.final_metrics ?? {}, 'baselines'),
    'random_review',
  );
  const expectedRandomFailures = numberAt(randomReview, 'expected_true_positives');
  const curves = objectAt(report, 'curves');
  const capacity = objectAt(curves, 'capacity');
  const fractions = arrayAt(capacity, 'fractions');
  return (
    <>
      <div className="page-heading">
        <div>
          <span className="eyebrow">RESEARCH WORKSPACE</span>
          <h1>Quality, with evidence.</h1>
          <p>A reproducible study of semiconductor failure prioritisation.</p>
        </div>
        <Badge tone="good">
          <span className="status-dot" /> Local workspace
        </Badge>
      </div>
      {finalAvailable && (
        <div className="outcome-banner" role="status">
          <Badge tone={experiments.final_status === 'supported' ? 'good' : 'warning'}>
            Final finding: {experiments.final_status.replaceAll('_', ' ')}
          </Badge>
          <p>
            The frozen policy found{' '}
            <strong>
              {format(numberAt(primary, 'true_positives'))} of{' '}
              {format(numberAt(primary, 'n_failures'))} failures
            </strong>{' '}
            while reviewing {format(numberAt(primary, 'review_count'))} observations.
            {expectedRandomFailures !== null && (
              <>
                {' '}
                Uniform random review would find {format(expectedRandomFailures, 2)} failures in
                expectation at the same capacity.
              </>
            )}
          </p>
        </div>
      )}
      <div className="stats-grid">
        <Stat
          label="Production records"
          value={format(numberAt(audit, 'n_samples'))}
          note="Historical SECOM observations"
        />
        <Stat
          label="Anonymous sensors"
          value={format(numberAt(audit, 'n_features'))}
          note="Feature availability is not established"
        />
        <Stat
          label="Observed failures"
          value={format(numberAt(classes, 'fail'))}
          note={`${percent((numberAt(classes, 'fail') ?? 0) / (numberAt(audit, 'n_samples') ?? 1))} of supplied observations`}
        />
        <Stat
          label="Inspection capacity"
          value="10%"
          note="Fixed research scenario · top-k ranking"
          accent
        />
      </div>
      <div className="overview-grid">
        <Panel title="The research question" subtitle="DECISION STUDY" className="question-panel">
          <div className="question-mark">
            <FlaskConical size={26} />
          </div>
          <h3>
            Can we find more failures <br />
            with a limited review budget?
          </h3>
          <p>
            Rank historical observations, compare simple models, and evaluate the selected policy on
            a later partition kept outside development.
          </p>
          <button className="button" onClick={() => navigate('experiments')}>
            Explore the evidence <ArrowRight size={16} />
          </button>
          <div className="question-footer">
            <LockKeyhole size={14} /> Final rows are excluded from interactive exploration.
          </div>
        </Panel>
        <Panel title="Study protocol" subtitle="A fixed route from data to evidence">
          <div className="protocol-step">
            <span>01</span>
            <div>
              <h3>Audit the source</h3>
              <p>Checksummed data, anonymous sensors and explicit missingness.</p>
            </div>
            <CheckCircle2 size={17} />
          </div>
          <div className="protocol-step">
            <span>02</span>
            <div>
              <h3>Select in development</h3>
              <p>Expanding forward folds. Preprocessing stays inside training.</p>
            </div>
            <GitBranch size={17} />
          </div>
          <div className="protocol-step">
            <span>03</span>
            <div>
              <h3>Freeze and evaluate</h3>
              <p>A later holdout, a fixed review policy and uncertainty.</p>
            </div>
            <LockKeyhole size={17} />
          </div>
          <div className="protocol-step">
            <span>04</span>
            <div>
              <h3>Inspect the limits</h3>
              <p>Retrospective results, separate demo replay, no factory claims.</p>
            </div>
            <Layers3 size={17} />
          </div>
        </Panel>
      </div>
      <div className="two-col">
        <Panel
          title="Holdout evidence"
          subtitle="Frozen aggregate results"
          action={
            <Badge tone={experiments.final_status === 'supported' ? 'good' : 'warning'}>
              {experiments.final_status.replaceAll('_', ' ')}
            </Badge>
          }
        >
          {finalAvailable ? (
            <>
              <div className="result-number">
                {percent(numberAt(primary, 'recall'))}
                <span>failure recall at the primary capacity</span>
              </div>
              <MetricTable
                rows={[
                  {
                    label: 'Reviewed observations',
                    value: `${format(numberAt(primary, 'review_count'))} / ${format(numberAt(primary, 'n_samples'))}`,
                  },
                  {
                    label: 'Failures found',
                    value: `${format(numberAt(primary, 'true_positives'))} / ${format(numberAt(primary, 'n_failures'))}`,
                  },
                  {
                    label: 'Average precision',
                    value: format(
                      numberAt(experiments.final_metrics ?? {}, 'average_precision'),
                      3,
                    ),
                  },
                  {
                    label: 'Precision among reviewed',
                    value: percent(numberAt(primary, 'precision')),
                  },
                ]}
              />
            </>
          ) : (
            <Empty title="Final evaluation is pending">
              Development evidence remains available. No final performance has been generated for
              this view.
            </Empty>
          )}
        </Panel>
        <Panel title="Data composition" subtitle="Observed labels · not model predictions">
          <Chart
            label="Counts of pass and fail observations"
            data={[
              {
                type: 'bar',
                x: ['Pass', 'Fail'],
                y: [numberAt(classes, 'pass') ?? 0, numberAt(classes, 'fail') ?? 0],
                marker: { color: ['#a9c9c1', '#d29b70'] },
                width: 0.45,
                hovertemplate: '%{x}: %{y}<extra></extra>',
              },
            ]}
            layout={{
              height: 230,
              yaxis: { title: { text: 'Observations' }, gridcolor: '#edf0f4' },
              margin: { l: 55, r: 20, t: 10, b: 35 },
            }}
          />
          <p className="panel-caption">
            Rare outcomes make small score changes uncertain. Report event counts alongside every
            metric.
          </p>
        </Panel>
      </div>
      {fractions.length > 0 && (
        <Panel
          title="Capacity sensitivity"
          subtitle="Frozen evaluation · scenario sensitivity, not a policy slider"
        >
          <Chart
            label="Failure recall at different review capacities"
            data={[
              {
                type: 'scatter',
                mode: 'lines+markers',
                x: fractions as number[],
                y: arrayAt(capacity, 'recall') as number[],
                line: { color: '#277464', width: 3 },
                marker: { size: 6 },
                name: 'Failure recall',
              },
            ]}
            layout={{
              xaxis: {
                title: { text: 'Review fraction' },
                tickformat: '.0%',
                gridcolor: '#edf0f4',
              },
              yaxis: { title: { text: 'Recall' }, tickformat: '.0%', gridcolor: '#edf0f4' },
            }}
          />
        </Panel>
      )}
      <Note>{model.limitation}</Note>
    </>
  );
}

export function Experiments({ data }: { data: ResearchData }) {
  const { experiments, report } = data;
  const rows = experiments.fold_results.filter(
    (row): row is JsonObject => !!row && typeof row === 'object' && !Array.isArray(row),
  );
  const selected = textAt(experiments.selection, 'config_id', 'None selected');
  const pr = objectAt(objectAt(report, 'curves'), 'precision_recall');
  return (
    <>
      <div className="page-heading">
        <div>
          <span className="eyebrow">FROZEN PROTOCOL</span>
          <h1>Experiments</h1>
          <p>Simple baselines, chronological validation and a traceable selection.</p>
        </div>
        <Badge>
          <FlaskConical size={13} /> Development only
        </Badge>
      </div>
      <div className="selection-banner">
        <GitBranch size={22} />
        <div>
          <span className="eyebrow">SELECTED CONFIGURATION</span>
          <h3>{selected}</h3>
          <p>
            {textAt(
              experiments.selection,
              'selection_rule',
              'Selection is recorded in the research report.',
            )}
          </p>
        </div>
        <Badge tone="good">Frozen selection</Badge>
      </div>
      <Panel
        title="Development comparisons"
        subtitle="Each row is one configuration and validation fold"
      >
        {rows.length ? (
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Configuration</th>
                  <th>Family</th>
                  <th>Fold</th>
                  <th>Validation failures</th>
                  <th>Recall at capacity</th>
                  <th>Average precision</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row, i) => (
                  <tr
                    key={`${textAt(row, 'config_id')}-${i}`}
                    className={row.config_id === selected ? 'selected-row' : ''}
                  >
                    <td>
                      <span className="table-strong">{textAt(row, 'config_id')}</span>
                      {row.config_id === selected && <span className="small-text">Selected</span>}
                    </td>
                    <td>{textAt(row, 'family')}</td>
                    <td>{String(row.fold ?? row.fold_index ?? '—')}</td>
                    <td>{format(numberAt(row, 'n_failures'))}</td>
                    <td>{percent(numberAt(row, 'recall_at_capacity'))}</td>
                    <td>{format(numberAt(row, 'average_precision'), 3)}</td>
                    <td>
                      <Badge>{textAt(row, 'status', 'recorded')}</Badge>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <Empty title="No experiment records">
            Run the reproducible development study to populate this view.
          </Empty>
        )}
      </Panel>
      <div className="two-col">
        <Panel title="Final evaluation" subtitle="Aggregate evidence; no final-row exploration">
          <MetricTable
            rows={[
              {
                label: 'Scientific finding',
                value: (
                  <Badge tone={experiments.final_status === 'supported' ? 'good' : 'warning'}>
                    {experiments.final_status.replaceAll('_', ' ')}
                  </Badge>
                ),
              },
              { label: 'Selected configuration', value: selected },
              { label: 'Primary review policy', value: 'Top 10% of eligible observations' },
              { label: 'Secondary scenario', value: 'Top 20%, declared sensitivity' },
              { label: 'Output interpretation', value: 'Uncalibrated ranking score' },
            ]}
          />
          <Note>
            Inconclusive or negative findings remain valid outcomes. The holdout does not select a
            replacement model.
          </Note>
        </Panel>
        <Panel title="Precision–recall" subtitle="Frozen holdout curve">
          {arrayAt(pr, 'recall').length > 0 ? (
            <Chart
              label="Holdout precision recall curve"
              data={[
                {
                  type: 'scatter',
                  mode: 'lines',
                  x: arrayAt(pr, 'recall') as number[],
                  y: arrayAt(pr, 'precision') as number[],
                  line: { color: '#277464', width: 2 },
                  fill: 'tozeroy',
                  fillcolor: 'rgba(39,116,100,.07)',
                },
              ]}
              layout={{
                xaxis: { title: { text: 'Recall' }, tickformat: '.0%', gridcolor: '#edf0f4' },
                yaxis: { title: { text: 'Precision' }, tickformat: '.0%', gridcolor: '#edf0f4' },
              }}
            />
          ) : (
            <Empty title="Curve not available">
              The final curve appears only after offline evaluation.
            </Empty>
          )}
        </Panel>
      </div>
      <Panel title="Reproduction record" subtitle="Provenance attached to the selected experiment">
        <MetricTable
          rows={[
            {
              label: 'Selection fingerprint',
              value: <code>{textAt(experiments.selection, 'selection_hash')}</code>,
            },
            { label: 'Study limitation', value: experiments.limitation },
          ]}
        />
      </Panel>
    </>
  );
}

function displayJson(value: Json): string {
  return typeof value === 'object' ? JSON.stringify(value, null, 2) : String(value);
}
export function ModelCard({ data }: { data: ResearchData }) {
  const { model, audit, experiments } = data;
  return (
    <>
      <div className="page-heading">
        <div>
          <span className="eyebrow">IDENTITY & LIMITS</span>
          <h1>Model card</h1>
          <p>The bundle, decision policy and evidence behind every score.</p>
        </div>
        <Badge>Research demonstration</Badge>
      </div>
      <div className="two-col">
        <Panel title="Serving identity" subtitle="One immutable bundle">
          <MetricTable
            rows={[
              { label: 'Bundle', value: <code>{model.bundle_id}</code> },
              { label: 'Model', value: model.model_id },
              { label: 'Purpose', value: model.purpose },
              { label: 'Score kind', value: model.score_kind },
              { label: 'Final finding', value: experiments.final_status.replaceAll('_', ' ') },
            ]}
          />
        </Panel>
        <Panel title="Intended use" subtitle="Retrospective prioritisation">
          <div className="prose">
            <p>
              Explore whether historical anonymous measurements contain information useful for
              prioritising a limited inspection queue.
            </p>
            <p>
              Every score is tied to a model bundle and input fingerprint. A review recommendation
              is a queue decision, not a pass/fail certificate.
            </p>
            <Note>{model.limitation}</Note>
          </div>
        </Panel>
      </div>
      <Panel
        title="Decision policy"
        subtitle="Frozen capacity policy; no arbitrary threshold tuning"
      >
        <div className="key-grid">
          {Object.entries(model.policy).map(([key, value]) => (
            <div key={key}>
              <span className="small-label">{key.replaceAll('_', ' ')}</span>
              <code>{displayJson(value)}</code>
            </div>
          ))}
        </div>
      </Panel>
      <div className="two-col">
        <Panel title="Data and attribution" subtitle="Historical benchmark">
          <div className="prose">
            <h3>UCI SECOM · 2008</h3>
            <p>
              Michael McCann and Adrian Johnston. SECOM, UCI Machine Learning Repository. DOI:
              10.24432/C54305.
            </p>
            <p>
              Dataset: CC BY 4.0. Anonymous process measurements; no verified sensor physics or
              pre-outcome availability.
            </p>
            <a
              className="text-link"
              href="https://archive.ics.uci.edu/dataset/179/secom"
              target="_blank"
              rel="noreferrer"
            >
              View original source <ArrowRight size={14} />
            </a>
            <p>
              {textAt(
                audit,
                'metadata_discrepancy',
                'The downloaded file schema is audited independently of repository metadata.',
              )}
            </p>
          </div>
        </Panel>
        <Panel title="Interpretation checklist" subtitle="What this study can establish">
          <ul className="checklist">
            <li>
              <CheckCircle2 size={16} /> Reproducible evaluation on a fixed historical partition.
            </li>
            <li>
              <CheckCircle2 size={16} /> Missingness, class imbalance and inspection tradeoffs.
            </li>
            <li>
              <FileText size={16} /> Association rather than physical root cause.
            </li>
            <li>
              <FileText size={16} /> No demonstrated early-warning lead time.
            </li>
            <li>
              <FileText size={16} /> No validated factory or electrical-test deployment.
            </li>
          </ul>
        </Panel>
      </div>
    </>
  );
}
