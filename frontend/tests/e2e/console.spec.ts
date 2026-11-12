import { expect, test } from '@playwright/test';

test('research views retain their boundaries', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto('/');
  await expect(page.getByRole('heading', { name: 'Quality, with evidence.' })).toBeVisible();
  for (const [label, heading] of [
    ['Data audit', 'Data audit'],
    ['Experiments', 'Experiments'],
    ['Model card', 'Model card'],
    ['Monitoring', 'Monitoring'],
  ]) {
    await page.getByRole('button', { name: label, exact: true }).click();
    await expect(page.getByRole('heading', { name: heading, exact: true })).toBeVisible();
  }
  await page.getByRole('button', { name: 'New replay', exact: true }).click();
  await page.getByRole('button', { name: 'Step 20', exact: true }).click();
  await expect(
    page.getByText('20 / 558 observations · cursor checked on every advance'),
  ).toBeVisible();
  expect(errors).toEqual([]);
});

test('approved CSV becomes an inspection queue and exports', async ({ page, request }) => {
  const sample = await request.get('/api/v1/demo/inputs');
  expect(sample.ok()).toBeTruthy();
  await page.goto('/');
  await page.getByRole('button', { name: 'Inspection queue', exact: true }).click();
  await page
    .getByLabel('Choose CSV file')
    .setInputFiles({ name: 'development.csv', mimeType: 'text/csv', buffer: await sample.body() });
  await page.getByRole('button', { name: 'Validate and score', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Prioritised records' })).toBeVisible();
  await expect(page.getByRole('link', { name: 'Export queue' })).toHaveAttribute(
    'href',
    /\/api\/v1\/batches\/.+\/export/,
  );
  const inspect = page.getByRole('button', { name: /^Inspect / }).first();
  await expect(inspect).toBeVisible();
  await inspect.click();
  await expect(page.getByRole('dialog')).toContainText('Uncalibrated risk score');
  await page.getByRole('button', { name: 'Close record details' }).click();
});
