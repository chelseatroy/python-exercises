// Backend for the intro polls page in docs/. It lives in a Google Sheet's
// Apps Script project and is deployed as a web app; see the README for setup.
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

function doGet() {
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
