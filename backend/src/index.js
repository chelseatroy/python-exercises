// Backend for the intro polls page, the Session 7 Dunder Bingo page, and the
// Session 8 Lost Updates page in docs/.
// It's a Cloudflare Worker with a D1 database; see the README for setup.
//
// Polls, at /:
// GET  -> { ok, counts: { pollId: [votesForOption0, votesForOption1, ...] } }
// POST -> body {"poll", "voter", "choice"}; responds { ok, choice, counts }.
//         A voter who already voted on that poll in the last 24 hours gets
//         their original choice back instead of a second vote.
// Votes are rows of (time, poll, voter, choice) in the votes table. Only votes
// from the last 24 hours count, and older rows are deleted on the next vote.
//
// Dunder Bingo, at /bingo:
// GET  -> { ok, players, counts: { square: playersWhoLitIt } }, over the last hour.
//         players counts everyone who lit at least one square.
// POST -> body {"player", "square"} records that the player lit that square now;
//         body {"player", "clear": true} forgets the player's whole card.
//         Responds like GET.
// Squares are rows of (time, player, square) in the bingo table. Only the last
// hour counts, and older rows are deleted on the next write.
//
// Lost Updates, at /race: one shared counter that the class adds 1 to the way
// `counter += 1` does, in three steps: read it, add 1, write it back.
// GET  -> { ok, now, round, mode, ends, counter, holder, rounds, recent }.
//         holder is the first 8 characters of the lock holder's id, or null.
//         rounds summarizes the last few rounds; recent lists the current round's last taps.
// POST -> body {"action": "start", "mode": "race" | "lock"} starts a new 30-second round at 0.
//         body {"action": "acquire", "player"} takes the lock, in lock mode, if it's free.
//         body {"action": "read", "player"} returns { value }: the counter right now.
//         body {"action": "write", "player", "round", "read"} sets the counter to read + 1
//         and records the tap; in lock mode only the holder can write, and writing releases
//         the lock. Responds { written, before } plus the state, where before is what the
//         counter held just before this write.
// Taps are rows of (round, mode, time, player, read, before) in race_taps; rows older
// than three hours are deleted when a new round starts.

const VOTE_LIFETIME_MS = 24 * 60 * 60 * 1000;
const BINGO_LIFETIME_MS = 60 * 60 * 1000;
// The dunders on the bingo card, the only squares the board accepts.
const BINGO_SQUARES = new Set([
  "__abstractmethods__", "__call__", "__class__", "__delattr__", "__dict__", "__dir__",
  "__doc__", "__eq__", "__format__", "__ge__", "__getattribute__", "__gt__", "__hash__",
  "__init__", "__init_subclass__", "__le__", "__lt__", "__module__", "__ne__", "__new__",
  "__reduce__", "__reduce_ex__", "__repr__", "__setattr__", "__sizeof__", "__slots__",
  "__str__", "__subclasshook__", "__weakref__",
]);
// The pages re-fetch every 5 seconds, and D1's free plan limits rows read per
// day, so each Worker instance reuses what it last read for a few seconds.
// Writes refresh it, so a voter always sees their own vote right away.
const CACHE_MS = 3000;
// Lost Updates: how long a round runs, how long a lock lasts if its holder never writes,
// and how long rows are kept. Its board refreshes every second, so it caches for one.
const RACE_ROUND_MS = 30 * 1000;
const RACE_LEASE_MS = 4000;
const RACE_LIFETIME_MS = 3 * 60 * 60 * 1000;
const RACE_CACHE_MS = 1000;

export default {
  async fetch(request, env) {
    if (request.method === "OPTIONS") return new Response(null, { status: 204, headers: CORS_HEADERS });
    const url = new URL(request.url);
    const bingo = url.pathname === "/bingo";
    const race = url.pathname === "/race";
    if (url.pathname !== "/" && !bingo && !race) return json({ ok: false, error: "Not found." }, 404);

    if (request.method === "GET") {
      if (bingo) return json({ ok: true, ...(await cached("bingo", () => readBingo(env.DB))) });
      if (race) return json({ ok: true, ...(await cached("race", () => readRace(env.DB), RACE_CACHE_MS)) });
      return json({ ok: true, counts: await cached("counts", () => readCounts(env.DB)) });
    }
    if (request.method !== "POST") return json({ ok: false, error: "Method not allowed." }, 405);

    // The pages send JSON as text/plain, which skips the CORS preflight.
    let body;
    try {
      body = JSON.parse(await request.text());
    } catch (error) {
      return json({ ok: false, error: "Request body isn't JSON." });
    }
    if (!body || typeof body !== "object") return json({ ok: false, error: "Request body isn't JSON." });
    if (race) return racePost(env.DB, body);
    return bingo ? bingoPost(env.DB, body) : votePost(env.DB, body);
  },
};

async function votePost(db, body) {
  const poll = String(body.poll);
  const voter = String(body.voter);
  const choice = body.choice;
  if (!/^[a-z0-9-]{1,40}$/.test(poll) || !/^[A-Za-z0-9-]{8,40}$/.test(voter) ||
      !Number.isInteger(choice) || choice < 0 || choice >= 10) {
    return json({ ok: false, error: "Invalid vote." });
  }

  // A batch runs as one transaction. The primary key on (poll, voter) makes a
  // second vote a no-op, and the select returns whichever vote is on record.
  const now = Date.now();
  const [, , existing] = await db.batch([
    db.prepare("DELETE FROM votes WHERE time < ?").bind(now - VOTE_LIFETIME_MS),
    db.prepare("INSERT OR IGNORE INTO votes (time, poll, voter, choice) VALUES (?, ?, ?, ?)").bind(now, poll, voter, choice),
    db.prepare("SELECT choice FROM votes WHERE poll = ? AND voter = ?").bind(poll, voter),
  ]);
  const counts = await refresh("counts", () => readCounts(db));
  return json({ ok: true, choice: existing.results[0].choice, counts });
}

async function readCounts(db) {
  const { results } = await db
    .prepare("SELECT poll, choice, COUNT(*) AS n FROM votes WHERE time >= ? GROUP BY poll, choice")
    .bind(Date.now() - VOTE_LIFETIME_MS)
    .all();
  const counts = {};
  for (const { poll, choice, n } of results) {
    if (!counts[poll]) counts[poll] = [];
    while (counts[poll].length <= choice) counts[poll].push(0);
    counts[poll][choice] = n;
  }
  return counts;
}

// ---------- Dunder Bingo ----------

async function bingoPost(db, body) {
  const player = String(body.player);
  if (!/^[A-Za-z0-9-]{8,40}$/.test(player)) return json({ ok: false, error: "Invalid player." });
  const now = Date.now();
  const prune = db.prepare("DELETE FROM bingo WHERE time < ?").bind(now - BINGO_LIFETIME_MS);
  if (body.clear === true) {
    await db.batch([prune, db.prepare("DELETE FROM bingo WHERE player = ?").bind(player)]);
  } else {
    const square = String(body.square);
    if (!BINGO_SQUARES.has(square)) return json({ ok: false, error: "Invalid square." });
    await db.batch([
      prune,
      db.prepare("INSERT OR REPLACE INTO bingo (time, player, square) VALUES (?, ?, ?)").bind(now, player, square),
    ]);
  }
  return json({ ok: true, ...(await refresh("bingo", () => readBingo(db))) });
}

async function readBingo(db) {
  const since = Date.now() - BINGO_LIFETIME_MS;
  const [squares, people] = await db.batch([
    db.prepare("SELECT square, COUNT(*) AS n FROM bingo WHERE time >= ? GROUP BY square").bind(since),
    db.prepare("SELECT COUNT(DISTINCT player) AS n FROM bingo WHERE time >= ?").bind(since),
  ]);
  const counts = {};
  for (const { square, n } of squares.results) counts[square] = n;
  return { players: people.results[0].n, counts };
}

// ---------- Lost Updates ----------

async function racePost(db, body) {
  const now = Date.now();
  const action = body.action;

  if (action === "start") {
    if (body.mode !== "race" && body.mode !== "lock") return json({ ok: false, error: "Invalid mode." });
    await db.batch([
      db.prepare("DELETE FROM race_taps WHERE time < ?").bind(now - RACE_LIFETIME_MS),
      db.prepare("UPDATE race_state SET round = round + 1, mode = ?, ends = ?, counter = 0, holder = NULL, holder_until = 0 WHERE id = 1")
        .bind(body.mode, now + RACE_ROUND_MS),
    ]);
    return json({ ok: true, ...(await refresh("race", () => readRace(db))) });
  }

  const player = String(body.player);
  if (!/^[A-Za-z0-9-]{8,40}$/.test(player)) return json({ ok: false, error: "Invalid player." });

  if (action === "acquire") {
    // Only one row can change, and only if nobody holds an unexpired lock.
    const result = await db
      .prepare("UPDATE race_state SET holder = ?, holder_until = ? WHERE id = 1 AND mode = 'lock' AND ends > ? AND (holder IS NULL OR holder_until < ? OR holder = ?)")
      .bind(player, now + RACE_LEASE_MS, now, now, player)
      .run();
    return json({ ok: true, acquired: result.meta.changes === 1 });
  }

  // Reading and writing both need a live round and, in lock mode, the lock.
  const allowed = "ends > ? AND (mode = 'race' OR (holder = ? AND holder_until >= ?))";

  if (action === "read") {
    const row = await db.prepare(`SELECT round, counter FROM race_state WHERE id = 1 AND ${allowed}`).bind(now, player, now).first();
    if (!row) return json({ ok: true, allowed: false });
    return json({ ok: true, allowed: true, round: row.round, value: row.counter });
  }

  if (action === "write") {
    const round = body.round, read = body.read;
    if (!Number.isInteger(round) || !Number.isInteger(read) || read < 0) return json({ ok: false, error: "Invalid write." });
    // One transaction: record the tap with what the counter held just before, then
    // overwrite the counter with read + 1, whatever it holds now. That overwrite is the bug.
    const where = `id = 1 AND round = ? AND ${allowed}`;
    const [tap] = await db.batch([
      db.prepare(`INSERT INTO race_taps (round, mode, time, player, read, before) SELECT round, mode, ?, ?, ?, counter FROM race_state WHERE ${where} RETURNING before`)
        .bind(now, player, read, round, now, player, now),
      db.prepare(`UPDATE race_state SET counter = ?, holder = CASE WHEN mode = 'lock' THEN NULL ELSE holder END WHERE ${where}`)
        .bind(read + 1, round, now, player, now),
    ]);
    const mine = tap.results[0];
    return json({ ok: true, written: !!mine, before: mine ? mine.before : null, ...(await refresh("race", () => readRace(db))) });
  }

  return json({ ok: false, error: "Invalid action." });
}

async function readRace(db) {
  const state = await db.prepare("SELECT round, mode, ends, counter, holder, holder_until FROM race_state WHERE id = 1").first();
  const [rounds, recent] = await db.batch([
    db.prepare(`SELECT round, mode, COUNT(*) AS taps, COUNT(DISTINCT player) AS players,
                       SUM(CASE WHEN before = read THEN 0 ELSE 1 END) AS clashes,
                       COUNT(*) - SUM(before - read) AS counter, MIN(time) AS first, MAX(time) AS last
                FROM race_taps GROUP BY round ORDER BY round DESC LIMIT 6`),
    db.prepare("SELECT time, substr(player, 1, 8) AS p, read, before FROM race_taps WHERE round = ? ORDER BY time DESC, rowid DESC LIMIT 60").bind(state.round),
  ]);
  const now = Date.now();
  const held = state.holder && state.holder_until >= now ? state.holder.slice(0, 8) : null;
  return {
    now, round: state.round, mode: state.mode, ends: state.ends, counter: state.counter, holder: held,
    rounds: rounds.results, recent: recent.results,
  };
}

// ---------- caching ----------

const memory = new Map();

async function cached(key, read, ms = CACHE_MS) {
  const hit = memory.get(key);
  if (hit && Date.now() - hit.at < ms) return hit.value;
  return refresh(key, read);
}

async function refresh(key, read) {
  const value = await read();
  memory.set(key, { at: Date.now(), value });
  return value;
}

// ---------- responses ----------

const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
};

function json(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json", "Cache-Control": "no-store", ...CORS_HEADERS },
  });
}
