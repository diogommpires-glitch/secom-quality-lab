import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, expect, it, vi } from 'vitest';
import Vision from '../pages/Vision';

const fetchMock = vi.fn<typeof fetch>();
function response(body: unknown, status = 200) {
  return Promise.resolve(
    new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } }),
  );
}
beforeEach(() => {
  fetchMock.mockReset();
  vi.stubGlobal('fetch', fetchMock);
});

it('shows honest unavailable state without scoring controls', async () => {
  fetchMock.mockImplementation((url) =>
    response(
      String(url).endsWith('/status')
        ? { ready: false, message: 'No trained bundle.' }
        : { batches: [] },
      String(url).endsWith('/status') ? 503 : 200,
    ),
  );
  render(<Vision />);
  expect(await screen.findByText('Vision research is not released')).toBeInTheDocument();
  expect(screen.queryByText('Validate and score')).not.toBeInTheDocument();
});

it('uploads images and uses backend queue counts and localisation', async () => {
  const prediction = {
    prediction_id: 'p1',
    record_id: 'development-item',
    risk_score: 0.4,
    rank: 1,
    review_recommended: true,
    quality_status: 'valid',
    quality_reason_codes: [],
    image_url: '/image.png',
    overlay_url: '/overlay.png',
    width: 24,
    height: 32,
    bundle_id: 'bundle-test',
  };
  const batch = {
    batch_id: 'batch-test',
    row_count: 10,
    eligible_count: 10,
    abstain_count: 0,
    review_capacity: 1,
    bundle_id: 'bundle-test',
    predictions: [prediction],
  };
  fetchMock.mockImplementation((url, init) =>
    response(
      String(url).endsWith('/status')
        ? { ready: true, manifest: { bundle_id: 'bundle-test' }, report: {} }
        : init?.method === 'POST'
          ? batch
          : { batches: [] },
    ),
  );
  render(<Vision />);
  await screen.findByText('Validate and score');
  fireEvent.change(screen.getByLabelText('Image batch'), {
    target: { files: [new File(['image'], 'batch.zip', { type: 'application/zip' })] },
  });
  fireEvent.click(screen.getByText('Validate and score'));
  await screen.findByText('development-item');
  expect(screen.getByText('Exact capacity across eligible images')).toBeInTheDocument();
  fireEvent.click(screen.getByText('development-item'));
  expect(await screen.findByAltText('Model localisation for development-item')).toHaveAttribute(
    'src',
    '/overlay.png',
  );
  await waitFor(() =>
    expect(
      fetchMock.mock.calls.some(
        ([, init]) => init?.method === 'POST' && init.body instanceof FormData,
      ),
    ).toBe(true),
  );
});
