// Backend for the intro polls page and the list resize board in docs/. It lives
// in a Google Sheet's Apps Script project and is deployed as a web app; see the
// README for setup. The resize board's requests carry board=resize and are
// handled at the bottom of this file.
//
// GET  -> { ok, counts: { pollId: [votesForOption0, votesForOption1, ...] } }
// POST -> body {"poll", "voter", "choice"}; responds { ok, choice, counts }.
//         A voter who already voted on that poll in the last 24 hours gets
//         their original choice back instead of a second vote.
//
// Votes are rows of (time, poll, voter, choice) on the "votes" tab. Only votes
// from the last 24 hours count, and older rows are deleted on the next vote.

const VOTE_LIFETIME_MS = 24 * 60 * 60 * 1000;
const SHEET_NAME = "votes";
const COUNTS_CACHE_KEY = "counts";
// Every vote refreshes the cached counts, so this only bounds how late a
// vote's 24-hour expiry can show up in the results.
const COUNTS_CACHE_SECONDS = 60;

function doGet(e) {
  if (e && e.parameter && e.parameter.board === "resize") return resizeGet();

  const cache = CacheService.getScriptCache();
  const cached = cache.get(COUNTS_CACHE_KEY);
  if (cached) return json({ ok: true, counts: JSON.parse(cached) });

  // Hold the lock so a vote can't land between reading the sheet and
  // caching, which would cache counts that are missing that vote.
  return withLock(() => json({ ok: true, counts: cacheCounts(recentRows(getSheet())) }));
}

function doPost(e) {
  let body;
  try {
    body = JSON.parse(e.postData.contents);
  } catch (error) {
    return json({ ok: false, error: "Request body isn't JSON." });
  }
  if (body && body.board === "resize") return resizePost(body);

  const poll = String(body.poll);
  const voter = String(body.voter);
  const choice = body.choice;
  if (!/^[a-z0-9-]{1,40}$/.test(poll) || !/^[A-Za-z0-9-]{8,40}$/.test(voter) ||
      !Number.isInteger(choice) || choice < 0 || choice >= 10) {
    return json({ ok: false, error: "Invalid vote." });
  }

  return withLock(() => {
    const sheet = getSheet();
    deleteExpiredRows(sheet);
    const rows = recentRows(sheet);
    const existing = rows.find((row) => row.poll === poll && row.voter === voter);
    if (existing) return json({ ok: true, choice: existing.choice, counts: cacheCounts(rows) });

    const time = new Date();
    sheet.appendRow([time, poll, voter, choice]);
    rows.push({ time: time.getTime(), poll, voter, choice });
    return json({ ok: true, choice, counts: cacheCounts(rows) });
  });
}

function withLock(fn) {
  const lock = LockService.getScriptLock();
  lock.waitLock(10000);
  try {
    return fn();
  } finally {
    lock.releaseLock();
  }
}

function getSheet() {
  const spreadsheet = SpreadsheetApp.getActiveSpreadsheet();
  let sheet = spreadsheet.getSheetByName(SHEET_NAME);
  if (!sheet) {
    sheet = spreadsheet.insertSheet(SHEET_NAME);
    sheet.appendRow(["time", "poll", "voter", "choice"]);
    // Keep ids like "257" as text rather than letting Sheets turn them into numbers.
    sheet.getRange("B:C").setNumberFormat("@");
  }
  return sheet;
}

function allRows(sheet) {
  return sheet.getDataRange().getValues().slice(1).map(([time, poll, voter, choice]) => ({
    time: new Date(time).getTime(),
    poll: String(poll),
    voter: String(voter),
    choice: Number(choice),
  }));
}

function recentRows(sheet) {
  const cutoff = Date.now() - VOTE_LIFETIME_MS;
  return allRows(sheet).filter((row) => row.time >= cutoff);
}

// Rows are appended in time order, so the expired ones are all at the top.
function deleteExpiredRows(sheet) {
  const cutoff = Date.now() - VOTE_LIFETIME_MS;
  const rows = allRows(sheet);
  let expired = 0;
  while (expired < rows.length && !(rows[expired].time >= cutoff)) expired += 1;
  if (expired > 0) sheet.deleteRows(2, expired);
}

function cacheCounts(rows) {
  const counts = {};
  for (const { poll, choice } of rows) {
    if (!counts[poll]) counts[poll] = [];
    while (counts[poll].length <= choice) counts[poll].push(0);
    counts[poll][choice] += 1;
  }
  CacheService.getScriptCache().put(COUNTS_CACHE_KEY, JSON.stringify(counts), COUNTS_CACHE_SECONDS);
  return counts;
}

function json(data) {
  return ContentService.createTextOutput(JSON.stringify(data)).setMimeType(ContentService.MimeType.JSON);
}

// ---------- List resize board (Session 3) ----------
//
// GET  ?board=resize -> { ok, submissions: [{ id, type, build, version, points }] }
// POST {"board": "resize", "voter", "type", "build", "version", "points"}
//        -> { ok, id, submissions }
//      points is [[length, bytes], ...], what the student measured with sys.getsizeof.
// POST {"board": "resize", "action": "delete", "voter", "id"} -> { ok, submissions }
//      Removes one of that voter's own submissions.
//
// Submissions are rows of (time, id, voter, type, build, version, points JSON) on
// the "resize" tab. Like votes, they count for 24 hours. Voter ids never leave the
// script, so nobody can delete someone else's submission.

const RESIZE_SHEET_NAME = "resize";
const RESIZE_CACHE_KEY = "resize-submissions";
const RESIZE_BUILDS = ["append", "literal", "comprehension", "list()", "extend", "repeat", "other"];
const RESIZE_MAX_PER_VOTER = 12;
const RESIZE_MAX_POINTS = 200;

function resizeGet() {
  const cached = CacheService.getScriptCache().get(RESIZE_CACHE_KEY);
  if (cached) return json({ ok: true, submissions: JSON.parse(cached) });
  return withLock(() => json({ ok: true, submissions: cacheSubmissions(recentSubmissions(getResizeSheet())) }));
}

function resizePost(body) {
  const voter = String(body.voter);
  if (!/^[A-Za-z0-9-]{8,40}$/.test(voter)) return json({ ok: false, error: "Invalid voter." });

  if (body.action === "delete") {
    const id = String(body.id);
    return withLock(() => {
      const sheet = getResizeSheet();
      const rows = allSubmissions(sheet);
      const index = rows.findIndex((row) => row.id === id && row.voter === voter);
      if (index >= 0) sheet.deleteRow(index + 2);
      return json({ ok: true, submissions: cacheSubmissions(recentSubmissions(sheet)) });
    });
  }

  const type = String(body.type || "").trim();
  const build = String(body.build);
  const version = String(body.version || "").trim();
  const points = body.points;
  if (type.length < 1 || type.length > 30) return json({ ok: false, error: "Describe the element type in 1-30 characters." });
  if (RESIZE_BUILDS.indexOf(build) < 0) return json({ ok: false, error: "Unknown build method." });
  if (!/^[0-9]\.[0-9]{1,2}(\.[0-9]{1,2})?[a-z0-9+]{0,8}$/.test(version)) return json({ ok: false, error: "Version should look like 3.12 or 3.12.4." });
  if (!Array.isArray(points) || points.length < 2 || points.length > RESIZE_MAX_POINTS ||
      !points.every((p) => Array.isArray(p) && p.length === 2 &&
        Number.isInteger(p[0]) && p[0] >= 0 && p[0] <= 100000 &&
        Number.isInteger(p[1]) && p[1] >= 0 && p[1] <= 100000000)) {
    return json({ ok: false, error: "Measurements should be 2-200 pairs of whole numbers." });
  }

  return withLock(() => {
    const sheet = getResizeSheet();
    deleteExpiredSubmissions(sheet);
    const rows = recentSubmissions(sheet);
    if (rows.filter((row) => row.voter === voter).length >= RESIZE_MAX_PER_VOTER) {
      return json({ ok: false, error: "That's the most submissions one person can make. Remove one first." });
    }
    const id = Utilities.getUuid().slice(0, 8);
    const time = new Date();
    sheet.appendRow([time, id, voter, type, build, version, JSON.stringify(points)]);
    rows.push({ time: time.getTime(), id, voter, type, build, version, points });
    return json({ ok: true, id, submissions: cacheSubmissions(rows) });
  });
}

function getResizeSheet() {
  const spreadsheet = SpreadsheetApp.getActiveSpreadsheet();
  let sheet = spreadsheet.getSheetByName(RESIZE_SHEET_NAME);
  if (!sheet) {
    sheet = spreadsheet.insertSheet(RESIZE_SHEET_NAME);
    sheet.appendRow(["time", "id", "voter", "type", "build", "version", "points"]);
    // Keep versions like "3.10" as text rather than letting Sheets turn them into 3.1.
    sheet.getRange("B:G").setNumberFormat("@");
  }
  return sheet;
}

function allSubmissions(sheet) {
  return sheet.getDataRange().getValues().slice(1).map(([time, id, voter, type, build, version, points]) => ({
    time: new Date(time).getTime(),
    id: String(id),
    voter: String(voter),
    type: String(type),
    build: String(build),
    version: String(version),
    points: JSON.parse(points),
  }));
}

function recentSubmissions(sheet) {
  const cutoff = Date.now() - VOTE_LIFETIME_MS;
  return allSubmissions(sheet).filter((row) => row.time >= cutoff);
}

function deleteExpiredSubmissions(sheet) {
  const cutoff = Date.now() - VOTE_LIFETIME_MS;
  const rows = allSubmissions(sheet);
  let expired = 0;
  while (expired < rows.length && !(rows[expired].time >= cutoff)) expired += 1;
  if (expired > 0) sheet.deleteRows(2, expired);
}

function cacheSubmissions(rows) {
  const submissions = rows.map(({ id, type, build, version, points }) => ({ id, type, build, version, points }));
  const text = JSON.stringify(submissions);
  // A cache entry holds at most 100KB. A full class fits easily; past that, skip the cache.
  if (text.length < 90000) {
    CacheService.getScriptCache().put(RESIZE_CACHE_KEY, text, COUNTS_CACHE_SECONDS);
  } else {
    CacheService.getScriptCache().remove(RESIZE_CACHE_KEY);
  }
  return submissions;
}
