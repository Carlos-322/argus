// The board page's script as served, run in a node vm under a stub DOM, its relative requests sent to a live board
// (or a static build) over real HTTP. render_smoke.js and viewer_nav.js drive the page through it.
//
//   const S = require('./page_sandbox')(PAGE.html, BASE_URL, {fetch, expose})
//
// fetch(url, real) may stand in for the page's fetch (real is the default one, to delay or count requests); expose
// is code appended to the page's script, to hand its internals to the driver. S.run() runs the page's script without
// starting it (the page starts itself with loadEpisodes(), which the driver calls). S.failure() is the first uncaught
// error, unhandled rejection or console error, S.failedRequests every request that did not succeed, S.popstate(url)
// and S.back() go to url or to the previous history entry and fire the page's popstate listeners, S.settle()
// waits until no request is in flight, and S.state() is what the page left on each element of its markup.
'use strict';
const fs = require('fs');
const vm = require('vm');

// an error that escapes any page counts against every page running in this process
const pages = new Set();
process.on('unhandledRejection', e => pages.forEach(fail => fail('unhandled rejection', e)));
process.on('uncaughtException', e => pages.forEach(fail => fail('uncaught exception', e)));

module.exports = function pageSandbox(pagePath, base, opts) {
  opts = opts || {};
  const html = fs.readFileSync(pagePath, 'utf8');
  const m = html.match(/<script>([\s\S]*?)<\/script>/);
  if (!m) throw new Error('no <script> block in ' + pagePath);
  // the page starts itself with its last statement; the driver starts it instead, once the run is instrumented
  const boot = '\nloadEpisodes();\n';
  if (!m[1].endsWith(boot)) throw new Error('the page does not end with loadEpisodes()');
  const pageJs = m[1].slice(0, -boot.length) + (opts.expose || '');
  // each element of the markup starts as its tag says: its attributes, classes and hidden
  const markup = new Map();
  for (const [, attrs] of html.slice(0, m.index).matchAll(/<[a-zA-Z][\w-]*((?:\s+[\w-]+(?:="[^"]*")?)*)\s*\/?>/g)) {
    const a = {};
    for (const [, k, v] of attrs.matchAll(/([\w-]+)(?:="([^"]*)")?/g)) a[k] = v === undefined ? '' : v;
    if (a.id) markup.set(a.id, a);
  }
  const markupIds = [...markup.keys()];

  let failure = null;
  const fail = (what, e) => { if (!failure) failure = what + ': ' + (e && e.stack || e); };
  pages.add(fail);

  const created = [];           // every element the page creates (its list cards among them)
  // ---- a stub DOM: every element takes any property and call, and keeps what the page writes into it ----
  const noop = () => {};
  function makeEl(id) {
    // attributes are kept, so a driver can compare what a page left on an element (S.state)
    const t = {id: id || '', _html: '', _text: '', _val: '', dataset: {}, children: [], hidden: false,
      _attrs: {}, setAttribute(k, v) { this._attrs[k] = String(v); }, removeAttribute(k) { delete this._attrs[k]; },
      getAttribute(k) { return k in this._attrs ? this._attrs[k] : null; }, hasAttribute(k) { return k in this._attrs; },
      style: {setProperty: noop, removeProperty: noop},
      classList: {_s: new Set(), add(...c) { c.forEach(x => this._s.add(x)); }, remove(...c) { c.forEach(x => this._s.delete(x)); },
        toggle(c, on) { const v = on === undefined ? !this._s.has(c) : !!on; if (v) this._s.add(c); else this._s.delete(c); return v; },
        contains(c) { return this._s.has(c); }}};
    for (const [k, v] of Object.entries(markup.get(id) || {})) {
      if (k === 'id') continue;
      if (k === 'class') v.split(/\s+/).filter(Boolean).forEach(c => t.classList._s.add(c));
      else if (k === 'hidden') t.hidden = true;
      else t._attrs[k] = v;
    }
    return new Proxy(t, {
      get(o, p) {
        if (p === '__target') return o;
        // a link's address and an element's title are its attributes, as in a browser
        if (p === 'href' || p === 'title') return p in o._attrs ? o._attrs[p] : '';
        if (p === 'innerHTML') return o._html;
        if (p === 'textContent') return o._text;
        if (p === 'value') return o._val;
        if (p in o) return o[p];
        switch (p) {
          case 'querySelectorAll': return () => [];
          case 'querySelector': return () => makeEl();
          case 'appendChild': case 'removeChild': case 'insertBefore': case 'append': return (c) => c;
          case 'closest': return () => null;
          case 'contains': return () => false;
          case 'cloneNode': return () => makeEl();
          case 'getContext': return () => new Proxy({}, {get: () => noop});
          case 'getBoundingClientRect': return () => ({top: 0, left: 0, width: 100, height: 100, right: 100, bottom: 100});
          case 'parentElement': case 'parentNode': case 'firstChild': case 'nextElementSibling': return null;
          case 'offsetTop': case 'offsetLeft': case 'offsetHeight': case 'offsetWidth': case 'scrollTop':
          case 'scrollLeft': case 'scrollWidth': case 'clientWidth': case 'clientHeight': case 'scrollHeight': return 0;
          case 'then': return undefined;
          default: return typeof p === 'string' ? noop : undefined;
        }
      },
      set(o, p, v) {
        if (p === 'href' || p === 'title') o._attrs[p] = String(v);
        else if (p === 'innerHTML') o._html = String(v);
        else if (p === 'textContent') o._text = String(v);
        else if (p === 'value') o._val = v;
        else o[p] = v;
        return true;
      },
    });
  }
  const byId = new Map();
  const doc = {
    getElementById: (id) => { if (!byId.has(id)) byId.set(id, makeEl(id)); return byId.get(id); },
    createElement: () => { const el = makeEl(); created.push(el); return el; },
    createDocumentFragment: () => makeEl(), querySelector: () => makeEl(), querySelectorAll: () => [],
    addEventListener: noop, removeEventListener: noop, body: makeEl('body'), documentElement: makeEl(),
    exitFullscreen: noop, fullscreenElement: null, activeElement: null,
  };
  // the sensors panel lays itself out only while its slot is on the page: the stub body holds that one element
  doc.body.contains = (el) => !!el && el.id === 'sn-slot';
  const store = new Map(), failedRequests = [], popstate = [];
  let inflight = 0, rafs = 0;
  const realFetch = (url, o) => fetch(new URL(String(url), base), {method: (o || {}).method,
    headers: (o || {}).headers, body: (o || {}).body});
  // the URL and the history entries the page's history calls leave, so a driver can go Back to an earlier one
  const location = {search: '', href: base, origin: new URL(base).origin, pathname: new URL(base).pathname,
    reload: noop};
  const setUrl = (u) => { const url = new URL(String(u), location.href); location.href = url.href;
    location.search = url.search; };
  const entries = [base];
  let at = 0;
  const ctx = {
    document: doc, console: {log: noop, warn: noop, info: noop, error: (...a) => fail('console.error', a.join(' '))},
    location,
    history: {
      replaceState: (s, t, u) => { if (u != null) { setUrl(u); entries[at] = location.href; } },
      pushState: (s, t, u) => { if (u != null) { setUrl(u); entries.splice(++at, entries.length, location.href); } },
    },
    localStorage: {getItem: k => store.has(k) ? store.get(k) : null, setItem: (k, v) => store.set(k, String(v)),
      removeItem: k => store.delete(k)},
    fetch: async (url, o) => {
      inflight++;
      try {
        const r = await (opts.fetch ? opts.fetch(url, () => realFetch(url, o)) : realFetch(url, o));
        // every request the page made that did not succeed, so a caller can require that none failed
        if (!r.ok) failedRequests.push(r.status + ' ' + url);
        return r;
      } finally { inflight--; }
    },
    setTimeout, clearTimeout, setInterval: () => 0, clearInterval: noop, queueMicrotask, structuredClone,
    requestAnimationFrame: (cb) => { if (++rafs < 5000) return setTimeout(() => cb(performance.now()), 4); return 0; },
    cancelAnimationFrame: (h) => clearTimeout(h),
    getComputedStyle: () => ({overflowY: 'auto', getPropertyValue: () => ''}),
    matchMedia: () => ({matches: false, addEventListener: noop, addListener: noop}),
    addEventListener: (type, fn) => { if (type === 'popstate') popstate.push(fn); }, removeEventListener: noop,
    alert: (msg) => fail('alert', msg), performance,
    innerWidth: 1440, innerHeight: 900, devicePixelRatio: 1,
    AbortController, URL, URLSearchParams, TextEncoder, TextDecoder, Blob, JSON, Math, Date, Promise, atob,
    IntersectionObserver: function () { return {observe: noop, disconnect: noop, unobserve: noop}; },
    ResizeObserver: function () { return {observe: noop, disconnect: noop, unobserve: noop}; },
    CSS: {supports: () => true, escape: s => s},
  };
  ctx.window = ctx;
  ctx.globalThis = ctx;
  vm.createContext(ctx);

  return {
    ctx, doc, created, failedRequests, fail, failure: () => failure,
    // what the page has left on each element of its markup (every id outside its script): content, attributes,
    // classes, hidden and every other property it set; an element it never touched reads as a fresh one
    state() {
      const untouched = makeEl();
      const describe = o => {
        const props = {};
        for (const [k, v] of Object.entries(o)) {
          if (['id', '_html', '_text', '_val', 'dataset', 'children', 'hidden', '_attrs', 'style', 'classList']
            .includes(k) || typeof v === 'function' || v === undefined) continue;
          if (v !== untouched[k]) props[k] = v;     // a scroll position of 0 is where an element starts
        }
        const style = Object.fromEntries(Object.entries(o.style).filter(([, v]) => typeof v !== 'function'));
        return {html: o._html, text: o._text, value: o._val, hidden: !!o.hidden, attrs: {...o._attrs},
          classes: [...o.classList._s].sort(), dataset: {...o.dataset}, style, props};
      };
      const out = {};
      for (const id of markupIds) out[id] = describe((byId.has(id) ? byId.get(id) : makeEl(id)).__target);
      return out;
    },
    run() { try { vm.runInContext(pageJs, ctx, {filename: 'board-page.js'}); } catch (e) { fail('top level', e); } },
    // the browser's Back or Forward to url: the URL changes first, then the page hears popstate
    popstate(url) { setUrl(url); return Promise.all(popstate.map(fn => fn({}))); },
    // the browser's Back button: the previous history entry, when there is one
    back() {
      if (at === 0) return Promise.resolve();
      setUrl(entries[--at]);
      return Promise.all(popstate.map(fn => fn({})));
    },
    // wait until the page has no request in flight and its timers have had a turn
    async settle() {
      for (let quiet = 0; quiet < 3;) {
        await new Promise(r => setTimeout(r, 30));
        quiet = inflight ? 0 : quiet + 1;
      }
    },
  };
};
