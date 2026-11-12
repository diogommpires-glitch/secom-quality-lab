import { expect, test } from '@playwright/test';

test('untrained service shows unavailable state without sample predictions', async ({ page }) => {
  await page.goto('http://127.0.0.1:8871/');
  await page.getByLabel('Research task').selectOption('vision');
  await expect(page.getByRole('heading', { name: 'Visual inspection', exact: true })).toBeVisible();
  await expect(
    page.getByRole('heading', { name: 'Vision research is not released' }),
  ).toBeVisible();
  await expect(page.getByRole('button', { name: 'Validate and score' })).toHaveCount(0);
  await expect(page.getByRole('link', { name: 'KolektorSDD2 source' })).toHaveAttribute(
    'href',
    'https://www.vicos.si/resources/kolektorsdd2/',
  );
});

test('synthetic-only fixture traverses real upload, queue, heatmap and provenance endpoints', async ({
  page,
  request,
}) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  const inputs = await request.get('/__test__/inputs.zip');
  expect(inputs.ok()).toBeTruthy();
  await page.goto('/');
  await page.getByLabel('Research task').selectOption('vision');
  await expect(page.getByText('SYNTHETIC TEST FIXTURE ONLY', { exact: true })).toBeVisible();
  await page.getByLabel('Image batch').setInputFiles({
    name: 'synthetic-fixture.zip',
    mimeType: 'application/zip',
    buffer: await inputs.body(),
  });
  const scored = page.waitForResponse(
    (response) =>
      response.url().endsWith('/api/v2/vision/batches') && response.request().method() === 'POST',
  );
  await page.getByRole('button', { name: 'Validate and score', exact: true }).click();
  const response = await scored;
  expect(response.status()).toBe(200);
  const batch = await response.json();
  expect(batch.row_count).toBe(10);
  expect(batch.review_capacity).toBe(1);
  expect(batch.bundle_id).toBe('synthetic-browser-fixture');
  await expect(page.getByRole('heading', { name: 'Inspection queue', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'synthetic-item-09', exact: true }).click();
  const overlay = page.getByAltText('Model localisation for synthetic-item-09');
  await expect(overlay).toBeVisible();
  await expect(overlay).toHaveJSProperty('naturalWidth', 24);
  await expect(
    page.getByText('Model localisation · display scale normalised per image'),
  ).toBeVisible();
  const image = await request.get(batch.predictions[0].image_url);
  expect(image.headers()['content-type']).toBe('image/png');
  const exported = await request.get(`/api/v2/vision/batches/${batch.batch_id}/export`);
  expect(exported.ok()).toBeTruthy();
  expect(await exported.text()).toContain('synthetic-item-09');
  await expect(page.getByRole('link', { name: 'Export queue' })).toHaveAttribute(
    'href',
    `/api/v2/vision/batches/${batch.batch_id}/export`,
  );
  const stored = await request.get(`/api/v2/vision/batches/${batch.batch_id}`);
  expect((await stored.json()).predictions).toEqual(batch.predictions);
  await page.screenshot({ path: 'test-results/vision-synthetic-workflow.png', fullPage: true });
  expect(errors).toEqual([]);
});
