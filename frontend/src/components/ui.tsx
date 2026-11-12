import type { ReactNode } from 'react';
import { AlertCircle, ArrowUpRight, Database, LoaderCircle } from 'lucide-react';

export function Loading({ label = 'Loading research records…' }: { label?: string }) {
  return (
    <div className="state" role="status">
      <LoaderCircle className="spin" size={25} />
      <p>{label}</p>
    </div>
  );
}
export function ErrorState({ message, retry }: { message: string; retry?: () => void }) {
  return (
    <div className="state error-state" role="alert">
      <AlertCircle size={27} />
      <h3>Unable to load this view</h3>
      <p>{message}</p>
      {retry && (
        <button className="button" onClick={retry}>
          Try again
        </button>
      )}
    </div>
  );
}
export function Empty({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="state">
      <Database size={28} />
      <h3>{title}</h3>
      <p>{children}</p>
    </div>
  );
}
export function Panel({
  title,
  subtitle,
  action,
  children,
  className = '',
}: {
  title: string;
  subtitle?: string;
  action?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`panel ${className}`}>
      <div className="panel-head">
        <div>
          <h2>{title}</h2>
          {subtitle && <p>{subtitle}</p>}
        </div>
        {action}
      </div>
      {children}
    </section>
  );
}
export function Stat({
  label,
  value,
  note,
  accent = false,
  warning = false,
}: {
  label: string;
  value: ReactNode;
  note: string;
  accent?: boolean;
  warning?: boolean;
}) {
  const classes = ['stat'];
  if (warning) classes.push('stat-warning');
  else if (accent) classes.push('stat-accent');
  return (
    <div className={classes.join(' ')}>
      <span className="stat-label">{label}</span>
      <strong>{value}</strong>
      <span className="stat-note">{note}</span>
    </div>
  );
}
export function Badge({
  children,
  tone = 'neutral',
}: {
  children: ReactNode;
  tone?: 'neutral' | 'good' | 'warning' | 'danger';
}) {
  return <span className={`badge badge-${tone}`}>{children}</span>;
}
export function Note({ children }: { children: ReactNode }) {
  return (
    <div className="note">
      <AlertCircle size={16} />
      <span>{children}</span>
    </div>
  );
}
export function LinkButton({ href, children }: { href: string; children: ReactNode }) {
  return (
    <a className="button button-secondary" href={href}>
      {children}
      <ArrowUpRight size={15} />
    </a>
  );
}
export function format(value: number | null | undefined, digits = 0): string {
  return value == null || !Number.isFinite(value)
    ? '—'
    : new Intl.NumberFormat('en-GB', { maximumFractionDigits: digits }).format(value);
}
export function percent(value: number | null | undefined, digits = 1): string {
  return value == null ? '—' : `${format(value * 100, digits)}%`;
}
export function MetricTable({ rows }: { rows: { label: string; value: ReactNode }[] }) {
  return (
    <dl className="metric-table">
      {rows.map((row) => (
        <div key={row.label}>
          <dt>{row.label}</dt>
          <dd>{row.value}</dd>
        </div>
      ))}
    </dl>
  );
}
