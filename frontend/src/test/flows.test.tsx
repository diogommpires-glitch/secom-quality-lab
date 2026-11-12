import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { ApiError, request } from '../api';
import Inspection from '../pages/Inspection';
import Monitoring from '../pages/Monitoring';

vi.mock('../components/Chart', () => ({ default: () => <div aria-label="Chart" /> }));
const fetchMock = vi.fn<typeof fetch>();
function response(body: unknown, status = 200) {
  return Promise.resolve(
    new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } }),
  );
}
function sampleFile() {
  const file = new File(['record_id,sensor_000\ndev-1,1'], 'sample.csv', { type: 'text/csv' });
  Object.defineProperty(file, 'text', {
    value: () => Promise.resolve('record_id,sensor_000\ndev-1,1'),
  });
  return file;
}
beforeEach(() => {
  fetchMock.mockReset();
  vi.stubGlobal('fetch', fetchMock);
});

describe('API failures', () => {
  it('preserves row-indexed server validation issues', async () => {
    fetchMock.mockImplementation(() =>
      response(
        {
          detail: {
            message: 'Invalid batch',
            issues: [
              {
                row_index: 3,
                field: 'sensor_001',
                reason_code: 'invalid_numeric',
                message: 'Expected a number',
              },
            ],
          },
        },
        422,
      ),
    );
    await expect(request('/batches')).rejects.toMatchObject({
      message: 'Invalid batch',
      issues: [
        {
          row_index: 3,
          field: 'sensor_001',
          reason_code: 'invalid_numeric',
          message: 'Expected a number',
        },
      ],
    });
    await expect(request('/batches')).rejects.toBeInstanceOf(ApiError);
  });
});

describe('Inspection workflow', () => {
  it('surfaces validation failure without creating a queue', async () => {
    fetchMock.mockImplementation((url, init) =>
      String(url).endsWith('/batches') && init?.method === 'POST'
        ? response(
            {
              detail: {
                message: 'CSV validation failed',
                issues: [
                  {
                    row_index: 2,
                    field: 'sensor_007',
                    reason_code: 'invalid_numeric',
                    message: 'Expected a numeric value',
                  },
                ],
              },
            },
            422,
          )
        : response({ batches: [] }),
    );
    render(<Inspection />);
    fireEvent.change(screen.getByLabelText('Choose CSV file'), {
      target: { files: [sampleFile()] },
    });
    await userEvent.click(await screen.findByRole('button', { name: /Validate and score/ }));
    expect(await screen.findByRole('alert')).toHaveTextContent('CSV validation failed');
    expect(screen.getByRole('alert')).toHaveTextContent(
      'Row 2 · sensor_007: Expected a numeric value',
    );
    expect(screen.getByText('No batch selected')).toBeInTheDocument();
  });
  it('shows server capacity, scores, trace and batch export', async () => {
    const prediction = {
      prediction_id: 'p1',
      record_id: 'dev-1',
      risk_score: 0.42,
      calibrated_probability: null,
      quality_status: 'eligible',
      quality_reason_codes: [],
      missingness: 0.05,
      range_violation_count: 0,
      bundle_id: 'bundle-test',
      model_id: 'logistic',
      input_fingerprint: 'sha256-test',
      rank: 1,
      review_recommended: true,
    };
    const batch = {
      batch_id: 'batch-1',
      bundle_id: 'bundle-test',
      model_id: 'logistic',
      queue_policy_id: 'top10',
      eligible_count: 10,
      row_count: 10,
      review_capacity: 1,
      abstain_count: 0,
      predictions: [prediction],
      validation_issues: [],
      replayed: false,
    };
    fetchMock.mockImplementation((_url, init) =>
      init?.method === 'POST' ? response(batch) : response({ batches: [] }),
    );
    render(<Inspection />);
    fireEvent.change(screen.getByLabelText('Choose CSV file'), {
      target: { files: [sampleFile()] },
    });
    await userEvent.click(await screen.findByRole('button', { name: /Validate and score/ }));
    expect(await screen.findByText('Prioritised records')).toBeInTheDocument();
    expect(screen.getByText('0.42')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /Export queue/ })).toHaveAttribute(
      'href',
      '/api/v1/batches/batch-1/export',
    );
    await userEvent.click(screen.getByRole('button', { name: 'Inspect dev-1' }));
    expect(screen.getByRole('dialog')).toHaveTextContent('sha256-test');
    expect(screen.getByRole('dialog')).toHaveTextContent('Uncalibrated risk score');
  });
});

describe('Monitoring workflow', () => {
  it('flushes delayed outcomes after all observations have arrived', async () => {
    const flushing = {
      stream_id: 'flush-1',
      scenario: 'baseline',
      status: 'flushing',
      cursor: 558,
      total_records: 558,
      latest_window: null,
    };
    fetchMock.mockImplementation((url, init) => {
      const path = String(url);
      if (path.endsWith('/advance')) return response({ ...flushing, cursor: 578 });
      if (path.endsWith('/demo/replays') && init?.method === 'POST') return response(flushing);
      if (path.endsWith('/windows')) return response({ windows: [] });
      return response({ streams: [] });
    });
    render(<Monitoring />);
    await userEvent.click(screen.getByRole('button', { name: /New replay/ }));
    const flush = await screen.findByRole('button', { name: /Flush outcomes/ });
    expect(flush).toBeEnabled();
    await userEvent.click(flush);
    await waitFor(() =>
      expect(screen.getByRole('button', { name: /Flush outcomes/ })).toBeEnabled(),
    );
    expect(
      screen.getByText('558 / 558 observations · cursor checked on every advance'),
    ).toBeInTheDocument();
    const advanceCall = fetchMock.mock.calls.find(([url]) => String(url).endsWith('/advance'));
    expect(JSON.parse(advanceCall?.[1]?.body as string)).toEqual({
      expected_cursor: 558,
      steps: 20,
    });
  });
  it('advances with the expected cursor and keeps missing outcomes unavailable', async () => {
    const initial = {
      stream_id: 'stream-1',
      scenario: 'baseline',
      cursor: 0,
      total_records: 558,
      latest_window: null,
    };
    const stepped = {
      ...initial,
      cursor: 20,
      latest_window: {
        cursor: 20,
        arrived_count: 20,
        matured_label_count: 0,
        pending_label_count: 20,
        abstain_count: 0,
        feature_window: {
          count: 20,
          missingness: 0.05,
          reference_missingness: 0.04,
          missingness_delta: 0.01,
          score_ks: 0.08,
          warning: false,
          consecutive_warning_windows: 0,
        },
        performance_window: {
          count: 0,
          positive_count: 0,
          negative_count: 0,
          coverage: 0,
          metrics: null,
          status: 'insufficient_data',
          first_sequence: null,
          last_sequence: null,
        },
      },
    };
    fetchMock.mockImplementation((url, init) => {
      const path = String(url);
      if (path.endsWith('/advance')) return response(stepped);
      if (path.endsWith('/demo/replays') && init?.method === 'POST') return response(initial);
      if (path.endsWith('/windows')) return response({ windows: [stepped.latest_window] });
      return response({ streams: [] });
    });
    render(<Monitoring />);
    await userEvent.click(screen.getByRole('button', { name: /New replay/ }));
    await userEvent.click(await screen.findByRole('button', { name: /Step 20/ }));
    await waitFor(() =>
      expect(
        screen.getByText('20 / 558 observations · cursor checked on every advance'),
      ).toBeInTheDocument(),
    );
    const advanceCall = fetchMock.mock.calls.find(([url]) => String(url).endsWith('/advance'));
    expect(JSON.parse(advanceCall?.[1]?.body as string)).toEqual({ expected_cursor: 0, steps: 20 });
    expect(
      screen.getByText(
        'Insufficient matured outcomes for performance. Missing metrics are not zero error.',
      ),
    ).toBeInTheDocument();
    expect(screen.getByText('insufficient data')).toBeInTheDocument();
  });
});
