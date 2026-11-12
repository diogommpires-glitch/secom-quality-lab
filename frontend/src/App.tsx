import { useCallback, useEffect, useState } from 'react';
import {
  Activity,
  ArrowUpRight,
  ChevronRight,
  Cpu,
  Database,
  FlaskConical,
  LayoutDashboard,
  ListOrdered,
  ShieldCheck,
} from 'lucide-react';
import type { ExperimentData, JsonObject, ModelInfo } from './api';
import { request } from './api';
import { ErrorState, Loading } from './components/ui';
import DataAudit from './pages/DataAudit';
import Inspection from './pages/Inspection';
import Monitoring from './pages/Monitoring';
import { Experiments, ModelCard, Overview } from './pages/Research';
import type { ResearchData } from './pages/Research';
import Vision from './pages/Vision';

export type Page =
  | 'overview'
  | 'data'
  | 'experiments'
  | 'inspection'
  | 'monitoring'
  | 'model'
  | 'vision';
const navigation = [
  { id: 'overview', label: 'Overview', icon: LayoutDashboard },
  { id: 'data', label: 'Data audit', icon: Database },
  { id: 'experiments', label: 'Experiments', icon: FlaskConical },
  { id: 'inspection', label: 'Inspection queue', icon: ListOrdered },
  { id: 'monitoring', label: 'Monitoring', icon: Activity },
  { id: 'model', label: 'Model card', icon: ShieldCheck },
] as const;
export default function App() {
  const [page, setPage] = useState<Page>('overview');
  const [data, setData] = useState<ResearchData | null>(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);
  const load = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const [model, audit, experiments, report] = await Promise.all([
        request<ModelInfo>('/model'),
        request<JsonObject>('/data-audit'),
        request<ExperimentData>('/experiments'),
        request<JsonObject>('/study-report'),
      ]);
      setData({ model, audit, experiments, report });
    } catch (e) {
      setError(e instanceof Error ? e.message : 'The research service could not be reached.');
    } finally {
      setLoading(false);
    }
  }, []);
  useEffect(() => {
    void load();
  }, [load]);
  function navigate(next: Page) {
    setPage(next);
    window.scrollTo({ top: 0, behavior: 'instant' });
  }
  let content;
  if (page === 'vision') content = <Vision />;
  else if (page === 'inspection') content = <Inspection />;
  else if (page === 'monitoring') content = <Monitoring />;
  else if (loading) content = <Loading />;
  else if (error) content = <ErrorState message={error} retry={() => void load()} />;
  else if (data) {
    if (page === 'overview') content = <Overview data={data} navigate={navigate} />;
    else if (page === 'data') content = <DataAudit audit={data.audit} />;
    else if (page === 'experiments') content = <Experiments data={data} />;
    else content = <ModelCard data={data} />;
  }
  return (
    <div className="app-shell">
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark">
            <Cpu size={23} />
          </span>
          <div>
            <strong>
              SECOM<span>Quality Lab</span>
            </strong>
            <span className="brand-version">RESEARCH CONSOLE / 1.0</span>
          </div>
        </div>
        <div className="sidebar-section">WORKSPACE</div>
        <label className="task-selector">
          Research task
          <select
            aria-label="Research task"
            value={page === 'vision' ? 'vision' : 'sensors'}
            onChange={(event) => navigate(event.target.value === 'vision' ? 'vision' : 'overview')}
          >
            <option value="sensors">SECOM Sensors</option>
            <option value="vision">Visual Inspection</option>
          </select>
        </label>
        <nav aria-label="Main navigation">
          {navigation.map((item) => (
            <button
              key={item.id}
              className={`nav-item ${page === item.id ? 'active' : ''}`}
              aria-current={page === item.id ? 'page' : undefined}
              onClick={() => navigate(item.id)}
            >
              <item.icon size={18} />
              <span>{item.label}</span>
              {page === item.id && <ChevronRight className="nav-chevron" size={14} />}
            </button>
          ))}
        </nav>
        <div className="sidebar-study">
          <span className="small-label">STUDY BOUNDARY</span>
          <div className="boundary-visual">
            <i />
            <i />
            <i />
            <i />
            <i />
            <i />
            <i />
            <i />
            <i />
            <i />
          </div>
          <strong>
            {page === 'vision' ? 'Training images → locked test' : 'Development → holdout'}
          </strong>
          <p>
            Historical evidence.
            <br />
            Explicit limitations.
          </p>
          <a
            href={
              page === 'vision'
                ? 'https://www.vicos.si/resources/kolektorsdd2/'
                : 'https://archive.ics.uci.edu/dataset/179/secom'
            }
            target="_blank"
            rel="noreferrer"
          >
            {page === 'vision' ? 'KolektorSDD2 source' : 'UCI source record'}{' '}
            <ArrowUpRight size={13} />
          </a>
        </div>
        <div className="sidebar-footer">
          <span className={`status-dot ${error ? 'status-offline' : ''}`} />
          <div>
            <strong>{error ? 'Service unavailable' : 'Local research environment'}</strong>
            <span>Local workflow · no factory connection</span>
          </div>
        </div>
      </aside>
      <div className="main-shell">
        <header className="topbar">
          <div className="breadcrumb">
            Workspace <ChevronRight size={13} />
            <strong>
              {page === 'vision'
                ? 'Visual inspection'
                : navigation.find((item) => item.id === page)?.label}
            </strong>
          </div>
          <div className="topbar-right">
            <span className="research-tag">
              <FlaskConical size={13} /> Retrospective research
            </span>
            <span className="avatar" aria-label="Local user">
              DP
            </span>
          </div>
        </header>
        <main id="main" tabIndex={-1}>
          {content}
        </main>
        <footer className="app-footer">
          <span>SECOM Quality Lab</span>
          <span>Uncalibrated scores. Human review. Reproducible evidence.</span>
        </footer>
      </div>
    </div>
  );
}
