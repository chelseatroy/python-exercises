import { POLL_API_URL } from "../s1_polls/config.js";

// Matches the backend: submissions older than this don't count.
const LIFETIME_MS = 24 * 60 * 60 * 1000;
const REFRESH_MS = 5000;
const SERIES_SLOTS = 8;
const BUILD_LABELS = {
  append: "append in a loop",
  literal: "a literal",
  comprehension: "a comprehension",
  "list()": "list(…)",
  extend: "extend / +=",
  repeat: "[x] * n",
  other: "something else",
};

// ?demo fills the board with made-up results and saves nothing, for trying the page out.
const DEMO = new URLSearchParams(location.search).has("demo");
const configured = DEMO || !POLL_API_URL.includes("REPLACE_ME");

const $ = (id) => document.getElementById(id);
const banner = $("banner");

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [name, value] of Object.entries(attrs)) {
    if (name === "class") node.className = value;
    else if (name === "text") node.textContent = value;
    else if (name === "style") node.style.cssText = value;
    else node.setAttribute(name, value);
  }
  node.append(...children);
  return node;
}

function storage(key, fallback) {
  return {
    load() {
      try {
        const raw = localStorage.getItem(key);
        return raw === null ? fallback() : JSON.parse(raw);
      } catch {
        return fallback();
      }
    },
    save(value) {
      try { localStorage.setItem(key, JSON.stringify(value)); } catch {}
    },
  };
}

function getVoterId() {
  const store = storage("resize-board-voter-id", () => null);
  let id = store.load();
  if (typeof id !== "string") {
    id = crypto.randomUUID();
    store.save(id);
  }
  return id;
}
const voterId = getVoterId();

// This browser's own submissions, as [{ id, t }].
const mineStore = storage("resize-board-mine", () => []);
function loadMine() {
  const cutoff = Date.now() - LIFETIME_MS;
  const mine = mineStore.load();
  return Array.isArray(mine) ? mine.filter((m) => m && m.t >= cutoff) : [];
}

// ---------- talking to the backend ----------

async function callApi(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const data = await response.json();
  if (!data.ok) throw new Error(data.error || "Request failed");
  return data;
}

const demoBackend = DEMO ? makeDemoBackend() : null;

function fetchSubmissions() {
  if (DEMO) return demoBackend.get();
  return callApi(`${POLL_API_URL}?board=resize`, { cache: "no-store" });
}

// A plain-string body goes out as text/plain, which skips the CORS preflight
// request.
function postToBoard(body) {
  if (DEMO) return demoBackend.post(body);
  return callApi(POLL_API_URL, { method: "POST", body: JSON.stringify({ board: "resize", voter: voterId, ...body }) });
}

// ---------- parsing what students paste ----------

function parsePoints(text) {
  const numbers = (text.match(/-?\d+/g) || []).map(Number);
  if (numbers.length === 0) return { points: [], error: null };
  if (numbers.length % 2 === 1) return { points: [], error: `That's ${numbers.length} numbers. Each length needs a size after it, so there should be an even count.` };
  const byLength = new Map();
  for (let i = 0; i < numbers.length; i += 2) byLength.set(numbers[i], numbers[i + 1]);
  const points = [...byLength.entries()].sort((a, b) => a[0] - b[0]);
  if (points.some(([n, b]) => n < 0 || b < 0)) return { points: [], error: "Lengths and sizes can't be negative." };
  if (points.length < 2) return { points, error: "Add at least two lengths so there's something to compare." };
  if (points.length > 200) return { points: [], error: "That's more than 200 lengths. Send just the ones where the size changed." };
  // Sizes in bytes are always bigger than lengths; a list never has fewer than 56 bytes.
  if (points.every(([n, b]) => n > b)) return { points, error: "These look backwards: put the length first, then the size." };
  return { points, error: null };
}

// The lengths where the size changed, as [[length, bytes], ...], starting with the first point.
function jumps(points) {
  const out = [];
  let last = null;
  for (const [n, b] of points) {
    if (b !== last) out.push([n, b]);
    last = b;
  }
  return out;
}

// sys.getsizeof at length n, read off a submission as a step function.
function sizeAt(points, n) {
  let size;
  for (const [len, bytes] of points) {
    if (len > n) break;
    size = bytes;
  }
  return size;
}

function sameShape(a, b) {
  const lo = Math.max(a[0][0], b[0][0]);
  const hi = Math.min(a[a.length - 1][0], b[b.length - 1][0]);
  if (lo > hi) return false;
  for (let n = lo; n <= hi; n++) if (sizeAt(a, n) !== sizeAt(b, n)) return false;
  return true;
}

// Groups submissions that agree wherever they overlap. Longer measurements go first,
// so a short run joins the longest pattern it fits. Groups keep the order their
// earliest submission arrived in, so colors don't jump around as results come in.
function groupByPattern(subs) {
  const order = new Map(subs.map((s, i) => [s.id, i]));
  const bySpan = [...subs].sort((a, b) => span(b) - span(a) || order.get(a.id) - order.get(b.id));
  const groups = [];
  for (const sub of bySpan) {
    const group = groups.find((g) => sameShape(g.rep.points, sub.points));
    if (group) group.members.push(sub);
    else groups.push({ rep: sub, members: [sub] });
  }
  for (const g of groups) g.first = Math.min(...g.members.map((m) => order.get(m.id)));
  groups.sort((a, b) => a.first - b.first);
  groups.forEach((g, i) => { g.key = `pattern-${i}`; g.label = `Pattern ${String.fromCharCode(65 + (i % 26))}`; });
  return groups;
}
const span = (s) => s.points[s.points.length - 1][0] - s.points[0][0];

const normType = (t) => t.trim().toLowerCase().replace(/\s+/g, " ");
const minorVersion = (v) => (v.match(/^\d+\.\d+/) || [v])[0];

// ---------- state ----------

let submissions = [];
let colorBy = "pattern";
let range = "40";
let build = "append";

// Each entity keeps the color slot it first got, so a new submission never repaints old ones.
const slotMemory = { pattern: new Map(), type: new Map(), build: new Map(), version: new Map() };
function slotFor(dimension, key) {
  const memory = slotMemory[dimension];
  if (!memory.has(key)) memory.set(key, memory.size);
  const slot = memory.get(key);
  return slot < SERIES_SLOTS ? `var(--series-${slot + 1})` : "var(--series-other)";
}

// ---------- the form ----------

const form = $("form");
const statusEl = $("status");

for (const button of $("type-chips").querySelectorAll("button")) {
  button.addEventListener("click", () => { $("type").value = button.textContent; updatePreview(); });
}
segment("build", (v) => { build = v; });
segment("colorby", (v) => { colorBy = v; drawResults(); });
segment("range", (v) => { range = v; drawResults(); });

function segment(id, onPick) {
  const group = $(id);
  group.addEventListener("click", (event) => {
    const button = event.target.closest("button");
    if (!button) return;
    for (const b of group.querySelectorAll("button")) b.setAttribute("aria-pressed", String(b === button));
    onPick(button.dataset.v);
  });
}

$("points").addEventListener("input", updatePreview);

function updatePreview() {
  const preview = $("preview");
  const { points, error } = parsePoints($("points").value);
  const parts = [el("span", { class: "label", text: "Preview" })];
  if (!points.length && !error) {
    parts.push(el("p", { class: "muted", text: "Paste your measurements to check them before you submit." }));
  } else {
    if (error) parts.push(el("p", { class: "status bad", text: error }));
    if (points.length) {
      const j = jumps(points);
      parts.push(el("p", { text: `${points.length} lengths, from ${points[0][0]} to ${points[points.length - 1][0]}. The size changed ${j.length - 1} time${j.length === 2 ? "" : "s"}:` }));
      const line = el("p", { class: "jumps" });
      j.forEach(([n, b], i) => {
        if (i) line.append(" → ");
        line.append(el("b", { text: `len ${n}` }), `: ${b} B`);
      });
      parts.push(line);
    }
  }
  preview.replaceChildren(...parts);
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const type = $("type").value.trim();
  const version = $("version").value.trim().replace(/^python\s*/i, "");
  const { points, error } = parsePoints($("points").value);
  statusEl.className = "status bad";
  if (!type) { statusEl.textContent = "Say what you put in the list."; return; }
  if (!/^[0-9]\.[0-9]{1,2}(\.[0-9]{1,2})?[a-z0-9+]{0,8}$/.test(version)) { statusEl.textContent = "The version should look like 3.12 or 3.12.4."; return; }
  if (error || points.length < 2) { statusEl.textContent = error || "Paste your measurements first."; return; }

  statusEl.className = "status";
  statusEl.textContent = "Sending…";
  $("submit").disabled = true;
  try {
    const data = await postToBoard({ type, build, version, points });
    mineStore.save([...loadMine(), { id: data.id, t: Date.now() }]);
    setSubmissions(data.submissions);
    statusEl.textContent = "It's on the board. Try another type or build method if you have time.";
    $("points").value = "";
    updatePreview();
  } catch (err) {
    console.error(err);
    statusEl.className = "status bad";
    statusEl.textContent = err.message.startsWith("HTTP") || err.message === "Failed to fetch"
      ? "That didn't go through. Check your connection and try again."
      : err.message;
  }
  $("submit").disabled = !configured;
});

function drawMine() {
  const mine = loadMine();
  const mineIds = new Set(mine.map((m) => m.id));
  const rows = submissions.filter((s) => mineIds.has(s.id));
  $("mine-wrap").hidden = rows.length === 0;
  $("mine").replaceChildren(...rows.map((s) => {
    const remove = el("button", { type: "button", class: "btn", text: "Remove" });
    remove.addEventListener("click", async () => {
      remove.disabled = true;
      try {
        setSubmissions((await postToBoard({ action: "delete", id: s.id })).submissions);
        mineStore.save(loadMine().filter((m) => m.id !== s.id));
      } catch (err) {
        console.error(err);
        remove.disabled = false;
      }
    });
    return el("li", {}, el("span", { text: `${s.type} · ${BUILD_LABELS[s.build] || s.build} · Python ${s.version} · ${s.points.length} lengths` }), remove);
  }));
}

// ---------- results ----------

function setSubmissions(list) {
  submissions = Array.isArray(list) ? list.filter((s) => Array.isArray(s.points) && s.points.length >= 2) : [];
  drawMine();
  drawResults();
}

function entityOf(sub, groupOf) {
  if (colorBy === "pattern") { const g = groupOf.get(sub.id); return { key: g.key, label: g.label }; }
  if (colorBy === "type") { const k = normType(sub.type); return { key: k, label: k }; }
  if (colorBy === "build") return { key: sub.build, label: BUILD_LABELS[sub.build] || sub.build };
  const k = minorVersion(sub.version);
  return { key: k, label: `Python ${k}` };
}

function drawResults() {
  const groups = groupByPattern(submissions);
  const groupOf = new Map();
  for (const g of groups) for (const m of g.members) groupOf.set(m.id, g);
  // Assign pattern colors in group order before anything else asks for one.
  for (const g of groups) slotFor("pattern", g.key);

  drawSummary(groups);
  drawChart(groupOf);
  drawGroups(groups);
}

function drawSummary(groups) {
  const summary = $("summary");
  if (!submissions.length) {
    summary.textContent = configured ? "No results yet. Be the first." : "";
    return;
  }
  const types = new Set(submissions.map((s) => normType(s.type)));
  const n = submissions.length;
  let text = `${n} submission${n === 1 ? "" : "s"}, ${types.size} element type${types.size === 1 ? "" : "s"}, ${groups.length} distinct pattern${groups.length === 1 ? "" : "s"}.`;
  const biggest = groups.reduce((a, b) => (b.members.length > a.members.length ? b : a));
  const bigTypes = new Set(biggest.members.map((m) => normType(m.type)));
  if (bigTypes.size > 1) text += ` ${biggest.label} alone covers ${bigTypes.size} different element types.`;
  summary.textContent = text;
}

const chart = $("chart");
const tip = $("tip");
const W = 900, H = 380, M = { l: 62, r: 20, t: 16, b: 40 };
let chartState = null;

function drawChart(groupOf) {
  const maxLen = range === "all"
    ? Math.max(10, ...submissions.map((s) => s.points[s.points.length - 1][0]))
    : 40;
  const visible = submissions.filter((s) => s.points[0][0] <= maxLen);
  if (!visible.length) {
    chart.innerHTML = `<text class="empty" x="${W / 2}" y="${H / 2}" text-anchor="middle">${submissions.length ? "Nothing in this range." : "Results show up here as people submit."}</text>`;
    $("legend").replaceChildren();
    chartState = null;
    return;
  }
  const sizes = visible.flatMap((s) => s.points.filter(([n]) => n <= maxLen).map(([, b]) => b));
  const minY = Math.floor(Math.min(...sizes) / 16) * 16;
  const maxY = Math.ceil(Math.max(...sizes) / 16) * 16 || minY + 16;
  const x = (n) => M.l + (n / maxLen) * (W - M.l - M.r);
  const y = (b) => H - M.b - ((b - minY) / Math.max(1, maxY - minY)) * (H - M.t - M.b);

  const parts = [];
  for (const t of niceTicks(minY, maxY, 5)) {
    parts.push(`<line class="grid" x1="${M.l}" x2="${W - M.r}" y1="${y(t)}" y2="${y(t)}"/><text x="${M.l - 8}" y="${y(t) + 4}" text-anchor="end">${t}</text>`);
  }
  for (const t of niceTicks(0, maxLen, 8)) {
    parts.push(`<text x="${x(t)}" y="${H - M.b + 18}" text-anchor="middle">${t}</text>`);
  }
  parts.push(`<line class="axis" x1="${M.l}" x2="${W - M.r}" y1="${H - M.b}" y2="${H - M.b}"/>`);
  parts.push(`<text x="${(M.l + W - M.r) / 2}" y="${H - 4}" text-anchor="middle">len(your_list)</text>`);
  parts.push(`<text x="14" y="${(M.t + H - M.b) / 2}" text-anchor="middle" transform="rotate(-90 14 ${(M.t + H - M.b) / 2})">sys.getsizeof (bytes)</text>`);

  const mineIds = new Set(loadMine().map((m) => m.id));
  const entities = new Map();
  // Draw mine last so they sit on top.
  const ordered = [...visible].sort((a, b) => Number(mineIds.has(a.id)) - Number(mineIds.has(b.id)));
  for (const sub of ordered) {
    const ent = entityOf(sub, groupOf);
    const color = slotFor(colorBy, ent.key);
    if (!entities.has(ent.key)) entities.set(ent.key, { ...ent, color, count: 0 });
    entities.get(ent.key).count += 1;
    const pts = sub.points.filter(([n]) => n <= maxLen);
    let d = "";
    pts.forEach(([n, b], i) => {
      if (i === 0) d += `M${x(n)} ${y(b)}`;
      else d += ` H${x(n)} V${y(b)}`;
    });
    // Carry the last measured size to the edge of the window if the run went further.
    const last = sub.points[sub.points.length - 1][0];
    if (last > maxLen) d += ` H${x(maxLen)}`;
    parts.push(`<path class="series${mineIds.has(sub.id) ? " mine" : ""}" stroke="${color}" d="${d}"/>`);
  }
  parts.push(`<line class="cross" id="cross" x1="0" x2="0" y1="${M.t}" y2="${H - M.b}" visibility="hidden"/>`);
  parts.push(`<rect id="hit" x="${M.l}" y="${M.t}" width="${W - M.l - M.r}" height="${H - M.t - M.b}" fill="transparent"/>`);
  chart.innerHTML = parts.join("");

  const sortedEntities = [...entities.values()].sort((a, b) => slotIndex(a) - slotIndex(b));
  $("legend").replaceChildren(...sortedEntities.map((e) =>
    el("span", {}, el("i", { class: "sw", style: `background:${e.color}` }), e.label, " ", el("span", { class: "n", text: `(${e.count})` }))));

  chartState = { maxLen, x, visible, groupOf };
}
const slotIndex = (e) => slotMemory[colorBy].get(e.key);

function niceTicks(lo, hi, count) {
  const spanV = hi - lo || 1;
  const raw = spanV / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 4, 5, 8, 10].map((m) => m * mag).find((s) => s >= raw) || raw;
  const out = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + 1e-9; t += step) out.push(Math.round(t));
  return out;
}

chart.addEventListener("mousemove", (event) => {
  if (!chartState) return;
  const box = chart.getBoundingClientRect();
  const px = ((event.clientX - box.left) / box.width) * W;
  if (px < M.l || px > W - M.r) { hideTip(); return; }
  const n = Math.round(((px - M.l) / (W - M.l - M.r)) * chartState.maxLen);
  const cross = $("cross");
  cross.setAttribute("x1", chartState.x(n));
  cross.setAttribute("x2", chartState.x(n));
  cross.setAttribute("visibility", "visible");

  // One row per color: what size that entity's lines report at this length.
  const rows = new Map();
  for (const sub of chartState.visible) {
    const size = sizeAt(sub.points, n);
    if (size === undefined || n > sub.points[sub.points.length - 1][0]) continue;
    const ent = entityOf(sub, chartState.groupOf);
    const key = `${ent.key}|${size}`;
    if (!rows.has(key)) rows.set(key, { ent, size, count: 0 });
    rows.get(key).count += 1;
  }
  const list = [...rows.values()].sort((a, b) => b.count - a.count).slice(0, 8);
  tip.replaceChildren(
    el("div", { text: `len ${n}` }),
    ...(list.length ? list.map((r) => el("div", {},
      el("i", { class: "sw", style: `background:${slotFor(colorBy, r.ent.key)}` }),
      `${r.size} B · ${r.ent.label}${r.count > 1 ? ` ×${r.count}` : ""}`)) : [el("div", { text: "nobody measured this length" })]),
  );
  tip.hidden = false;
  const card = chart.parentElement.getBoundingClientRect();
  const left = event.clientX - card.left + 14;
  tip.style.left = `${Math.min(left, card.width - tip.offsetWidth - 8)}px`;
  tip.style.top = `${event.clientY - card.top + 14}px`;
});
chart.addEventListener("mouseleave", hideTip);
function hideTip() {
  tip.hidden = true;
  const cross = document.getElementById("cross");
  if (cross) cross.setAttribute("visibility", "hidden");
}

function tally(values) {
  const counts = new Map();
  for (const v of values) counts.set(v, (counts.get(v) || 0) + 1);
  return [...counts.entries()].sort((a, b) => b[1] - a[1]);
}

function drawGroups(groups) {
  $("groups").replaceChildren(...groups.map((g) => {
    const j = jumps(g.rep.points).filter(([n]) => n <= 100);
    const jumpLine = el("p", { class: "jumps" });
    j.forEach(([n, b], i) => {
      if (i) jumpLine.append(" → ");
      jumpLine.append(`${n}: ${b}`);
    });
    const facet = (label, values) => el("div", { class: "facet" },
      el("span", { class: "label", text: label }),
      el("div", { class: "tags" }, ...tally(values).map(([v, c]) => el("span", {}, v, c > 1 ? el("b", { text: ` ×${c}` }) : ""))));
    return el("article", { class: "card group" },
      el("h3", {}, el("i", { class: "sw", style: `background:${slotFor("pattern", g.key)}` }), `${g.label}: ${g.members.length} submission${g.members.length === 1 ? "" : "s"}`),
      el("span", { class: "label", text: "Length: bytes, wherever the size changed" }),
      jumpLine,
      facet("Element types", g.members.map((m) => normType(m.type))),
      facet("Built with", g.members.map((m) => BUILD_LABELS[m.build] || m.build)),
      facet("Python", g.members.map((m) => m.version)));
  }));
}

// ---------- refreshing ----------

let refreshing = false;
async function refresh() {
  if (refreshing) return;
  refreshing = true;
  try {
    setSubmissions((await fetchSubmissions()).submissions);
    if (!DEMO) banner.hidden = true;
  } catch (err) {
    console.error(err);
    showBanner("Couldn't load results. They'll retry in a few seconds.");
  }
  refreshing = false;
}

function showBanner(message) {
  banner.textContent = message;
  banner.hidden = false;
}

let timer = null;
function schedule() {
  const wanted = configured && !document.hidden;
  if (wanted && timer === null) timer = setInterval(refresh, REFRESH_MS);
  else if (!wanted && timer !== null) { clearInterval(timer); timer = null; }
}

if (!configured) {
  showBanner("The board isn't set up yet: docs/s1_polls/config.js still has a placeholder URL.");
  $("submit").disabled = true;
  drawResults();
} else {
  if (DEMO) showBanner("Demo mode: these results are made up, and nothing you submit is saved.");
  refresh();
  schedule();
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) refresh();
    schedule();
  });
}

// ---------- demo data ----------

function makeDemoBackend() {
  const grow = (rule, upTo) => {
    const points = [[0, 56]];
    let alloc = 0;
    for (let n = 1; n <= upTo; n++) {
      if (n > alloc) alloc = rule(n);
      points.push([n, 56 + 8 * alloc]);
    }
    return points;
  };
  const modern = (n) => (n + (n >> 3) + 6) & ~3;
  const old = (n) => n + (n >> 3) + (n < 9 ? 3 : 6);
  const exactEven = (upTo) => Array.from({ length: upTo + 1 }, (_, n) => [n, 56 + 8 * (n + (n & 1))]);
  const exact = (upTo) => Array.from({ length: upTo + 1 }, (_, n) => [n, 56 + 8 * n]);
  const rows = [
    ["int", "append", "3.12.4", grow(modern, 40)],
    ["str", "append", "3.13.1", grow(modern, 30)],
    ["None", "append", "3.11.8", grow(modern, 60)],
    ["a long str", "append", "3.12.2", grow(modern, 25)],
    ["list", "comprehension", "3.14.0", grow(modern, 40)],
    ["float", "append", "3.10.8", grow(modern, 20)],
    ["int", "append", "3.8.10", grow(old, 40)],
    ["int", "list()", "3.10.8", exactEven(30)],
    ["int", "literal", "3.12.4", exact(12)],
    ["dict", "append", "3.12.4", grow(modern, 50)],
  ];
  // Keep the jumps plus the last length measured, the way a student might paste them.
  const compact = (points) => {
    const j = jumps(points);
    const last = points[points.length - 1];
    return j[j.length - 1][0] === last[0] ? j : [...j, last];
  };
  let list = rows.map(([type, b, version, points], i) => ({ id: `demo${i}`, type, build: b, version, points: compact(points) }));
  let next = 100;
  return {
    async get() { return { ok: true, submissions: list }; },
    async post(body) {
      if (body.action === "delete") {
        list = list.filter((s) => s.id !== body.id);
        return { ok: true, submissions: list };
      }
      const id = `demo${next++}`;
      list = [...list, { id, type: body.type, build: body.build, version: body.version, points: body.points }];
      return { ok: true, id, submissions: list };
    },
  };
}
