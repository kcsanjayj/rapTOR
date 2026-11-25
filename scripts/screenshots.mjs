// Captures the README screenshots from a running server.
// Usage: npm i puppeteer-core @sparticuz/chromium && node scripts/screenshots.mjs http://127.0.0.1:8799 docs/screenshots
import chromium from '@sparticuz/chromium';
import puppeteer from 'puppeteer-core';
const [base = 'http://127.0.0.1:8787', out = 'docs/screenshots'] = process.argv.slice(2);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const browser = await puppeteer.launch({ args: [...chromium.args, '--no-sandbox'], executablePath: await chromium.executablePath(), headless: 'shell' });
const page = await browser.newPage();
await page.setViewport({ width: 1600, height: 960, deviceScaleFactor: 1 });
const shot = (n) => page.screenshot({ path: `${out}/${n}.png` });
const run = async (text, { faults = false, latency = 800 } = {}) => {
  await page.goto(base, { waitUntil: 'networkidle0' });
  await page.$eval('#req', (e, t) => (e.value = t), text);
  await page.$eval('#lat', (e, v) => { e.value = v; e.dispatchEvent(new Event('input')); }, String(latency));
  if (faults) { await page.click('#cfgbtn'); await page.click('#chaos'); await page.click('#cfgbtn'); }
  await page.click('#go');
};
const done = async () => { await page.waitForFunction(() => document.querySelector('#go').textContent === 'RUN' && document.querySelector('#stats b'), { timeout: 60000 }); await sleep(4500); };

await page.goto(base, { waitUntil: 'networkidle0' }); await sleep(900); await shot('01-idle');
await run('Compare Rust and Go for a CLI tool'); await sleep(2400); await page.click('.tab[data-t=tl]'); await sleep(300); await shot('02-running'); await done(); await page.click('.tab[data-t=ans]'); await sleep(4500); await shot('03-completed');
await run('Compare Rust and Go for a CLI tool', { faults: true, latency: 500 }); await sleep(3600); await page.click('.tab[data-t=log]'); await sleep(300); await shot('04-faults-and-repair'); await done();
await run('Calculate 9**9**9', { latency: 500 }); await done(); await shot('05-failure-handled');
await browser.close();
