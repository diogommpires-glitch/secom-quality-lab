import { useEffect, useRef } from 'react';
import type { Data, Layout } from 'plotly.js';

export default function Chart({
  data,
  layout = {},
  label,
}: {
  data: Data[];
  layout?: Partial<Layout>;
  label: string;
}) {
  const element = useRef<HTMLDivElement>(null);
  useEffect(() => {
    let stopped = false;
    const node = element.current;
    void import('plotly.js-basic-dist-min').then((plotly) => {
      if (!stopped && node)
        void plotly.react(
          node,
          data,
          {
            autosize: true,
            height: 285,
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: { family: 'Inter, system-ui, sans-serif', size: 11, color: '#637084' },
            margin: { l: 42, r: 18, t: 18, b: 40 },
            xaxis: { gridcolor: '#edf0f4', zeroline: false },
            yaxis: { gridcolor: '#edf0f4', zeroline: false },
            ...layout,
          },
          { responsive: true, displayModeBar: false },
        );
    });
    return () => {
      stopped = true;
      if (node) void import('plotly.js-basic-dist-min').then((plotly) => plotly.purge(node));
    };
  }, [data, layout]);
  return <div className="chart" ref={element} role="img" aria-label={label} />;
}
