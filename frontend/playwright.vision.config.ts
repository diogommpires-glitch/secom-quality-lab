import { defineConfig, devices } from '@playwright/test';
import { resolve } from 'node:path';

const python =
  process.env.SECOM_TEST_PYTHON ??
  resolve(
    process.cwd(),
    process.platform === 'win32' ? '../.venv/Scripts/python.exe' : '../.venv/bin/python',
  );
export default defineConfig({
  testDir: './tests/vision',
  fullyParallel: false,
  workers: 1,
  use: { baseURL: 'http://127.0.0.1:8870', trace: 'retain-on-failure' },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
  webServer: [
    {
      command: `"${python}" ../tests/vision_server.py --port 8870 --synthetic-ready`,
      url: 'http://127.0.0.1:8870/health/live',
      reuseExistingServer: false,
      timeout: 30000,
    },
    {
      command: `"${python}" ../tests/vision_server.py --port 8871`,
      url: 'http://127.0.0.1:8871/health/live',
      reuseExistingServer: false,
      timeout: 30000,
    },
  ],
  reporter: [['list']],
});
