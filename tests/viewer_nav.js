// The board page under navigation that outruns its requests, against a live board: a read that fails once must not
// stay failed for the rest of the visit, and a response that arrives after the reader moved on must not undo the move.
//
//   node tests/viewer_nav.js PAGE.html BASE_URL
//
// Each case runs the page afresh (page_sandbox.js) and reports what the reader would see. Prints a JSON summary and
// exits 1 on the first uncaught error, unhandled rejection or console error.
'use strict';
const [pagePath, base] = process.argv.slice(2);
const sandbox = require('./page_sandbox');
const wait = ms => new Promise(r => setTimeout(r, ms));
const expose = `
globalThis.__nav = {eps: () => ALL_EPS, cmp: () => CMP, active: () => _activeFile, by: () => BY, cmpHas,
                    dataset: () => currentDataset, rail: () => railEps().filter(e => datasetOf(e) === currentDataset),
                    drawn: () => { try { return _drawnFile; } catch (_) { return 'unknown'; } },
                    cached: f => _epCache.has(f)};
`;

// a page whose requests matching a rule can fail once or be held until released
function page(rules) {
  const P = sandbox(pagePath, base, {expose, fetch: async (url, real) => {
    for (const rule of rules) if (rule.match(String(url))) return rule.answer(real);
    return real();
  }});
  P.run();
  return P;
}
const shown = P => ({active: P.ctx.__nav.active(), drawn: P.ctx.__nav.drawn(), search: P.ctx.location.search,
  compare_view: P.doc.body.classList.contains('view-cmp'), fading: P.doc.body.classList.contains('view-fade'),
  error: P.doc.getElementById('right-col').innerHTML.includes('could not be loaded'),
  title: P.doc.getElementById('current-ep').textContent});

(async () => {
  const out = {};
  const failures = [];
  const done = P => { if (P.failure()) failures.push(P.failure()); };

  // 1. An episode whose read fails once (the server's 500 with a JSON error body, a document caught mid-write), then
  // is repaired: opening it again shows it.
  {
    let failNext = null;
    const P = page([{match: u => failNext && /api\/episode\?file=|\/ep\//.test(u) && u.includes(encodeURIComponent(failNext)),
      answer: async () => { failNext = null; return new Response(JSON.stringify({error: 'The saved document could not be read.'}),
        {status: 500, headers: {'Content-Type': 'application/json'}}); }}]);
    await P.ctx.loadEpisodes(); await P.settle();
    const [a, b] = P.ctx.__nav.eps();
    await P.ctx.setDataset(a.dataset, a.file); await P.settle();
    out.first = shown(P);
    failNext = b.file;
    await P.ctx.setDataset(b.dataset, b.file); await P.settle();
    out.failed_read = {...shown(P), cached: P.ctx.__nav.cached(b.file)};
    await P.ctx.setDataset(a.dataset, a.file); await P.settle();
    await P.ctx.setDataset(b.dataset, b.file); await P.settle();
    out.after_repair = {...shown(P), cached: P.ctx.__nav.cached(b.file), file: b.file};
    done(P);
  }

  // 2. The comparison opened, then Back pressed while its metrics are still on their way: the episode stays.
  for (const [name, holdMetrics, backAfter] of [['back_while_metrics_load', true, 50],
                                                 ['back_while_view_fades', false, 60],
                                                 ['no_back', true, null]]) {
    let release;
    const held = new Promise(r => { release = r; });
    const P = page([{match: u => holdMetrics && /compare\/metrics/.test(u),
      answer: async real => { await held; return real(); }}]);
    await P.ctx.loadEpisodes(); await P.settle();
    const [a] = P.ctx.__nav.eps();
    await P.ctx.setDataset(a.dataset, a.file); await P.settle();
    const before = P.ctx.location.search;
    const opening = P.ctx.showCompare();
    if (backAfter != null) { await wait(backAfter); await P.popstate(before); }
    await wait(20); release();
    await opening; await wait(400); await P.settle();
    out[name] = {...shown(P), back_to: before};
    done(P);
  }

  // 3. Random clicks, tabs, labellers, the comparison and Back, each made before the last one's responses arrive, and
  // every response late by a random time: once all have arrived, the page shows what its URL says (the comparison
  // exactly when the URL asks for it, otherwise the URL's episode drawn under the URL's labeller, and no fade left on).
  const rng = seed => () => {
    seed = seed + 0x6D2B79F5 | 0;
    let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
  out.random = {runs: 0, wrong: []};
  for (let seed = 1; seed <= Number(process.env.VIEWER_NAV_RUNS || 25); seed++) {
    const rand = rng(seed), pick = xs => xs[Math.floor(rand() * xs.length)];
    const P = page([{match: () => true, answer: async real => { await wait(rand() * 120); return real(); }}]);
    const N = () => P.ctx.__nav, inCompare = () => P.doc.body.classList.contains('view-cmp');
    let last = null;
    const renderEp = P.ctx.renderEp;
    P.ctx.renderEp = function (d, o) {
      last = {id: (d._meta || {}).episode_id, by: (d._compare || {}).key || null};
      return renderEp.call(this, d, o);
    };
    await P.ctx.loadEpisodes(); await P.settle(); await wait(300);
    const eps = N().eps(), keys = [null, ...((N().cmp() || {}).models || []).map(m => m.key)];
    const steps = [];
    // what a reader can do on the screen in front of them: the list's cards, the dataset tabs, the Labels by menu,
    // the comparison and its episode links, and the browser's Back
    const act = () => {
      const r = rand();
      if (r < 0.25 && !inCompare()) {
        const rail = N().rail();
        if (rail.length) { const e = pick(rail); steps.push('card ' + e.file); return P.ctx.selectEp(e.file); }
      }
      if (r < 0.4) {
        const ds = pick(eps).dataset;
        steps.push('tab ' + ds);
        return inCompare() ? P.ctx.leaveCompare(ds) : (ds !== N().dataset() ? P.ctx.setDataset(ds) : null);
      }
      if (r < 0.55) { const k = pick(keys); steps.push('labeller ' + k); return P.ctx.setLabeller(k); }
      if (r < 0.7) { steps.push('compare'); return P.ctx.showCompare(); }
      if (r < 0.8 && inCompare()) {
        const e = pick(eps), k = pick(keys);
        steps.push('table link ' + e.file + ' ' + k);
        return P.ctx.leaveCompare(null, e.file, k && N().cmpHas(k, e.file) ? k : null);
      }
      steps.push('back');
      return P.back();
    };
    for (let i = 0; i < 6; i++) { act(); await wait(rand() * 100); }
    await P.settle(); await wait(600); await P.settle();
    const q = new URLSearchParams(P.ctx.location.search), active = N().active();
    const want = {compare: q.get('view') === 'compare', ep: q.get('ep'), by: q.get('by') || null};
    const rec = eps.find(e => e.file === active) || {};
    const shows = {compare: inCompare(), fading: P.doc.body.classList.contains('view-fade')
      || P.doc.body.classList.contains('lb-swap'), active, by: N().by(), drawn: last};
    const right = shows.compare === want.compare && !shows.fading && (want.compare || (
      (!want.ep || want.ep === active) && shows.by === want.by && last && last.id === rec.episode_id
      && last.by === (shows.by && N().cmpHas(shows.by, active) ? shows.by : null)));
    out.random.runs++;
    if (!right) out.random.wrong.push({seed, steps, url: P.ctx.location.search, shows});
    done(P);
  }

  process.stdout.write(JSON.stringify(out) + '\n');
  if (failures.length) { console.error('VIEWER NAV FAIL\n' + failures.join('\n')); process.exit(1); }
  process.exit(0);
})();
