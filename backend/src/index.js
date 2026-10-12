// Backend for the intro polls page and the list resize board in docs/. It's a
// Cloudflare Worker with a D1 database; see the README for setup. The resize
// board's requests carry board=resize and are handled at the bottom of this file.
//
// GET  -> { ok, counts: { pollId: [votesForOption0, votesForOption1, ...] } }
// POST -> body {"poll", "voter", "choice"}; responds { ok, choice, counts }.
//         A voter who already voted on that poll in the last 24 hours gets
//         their original choice back instead of a second vote.
//
// Votes are rows of (time, poll, voter, choice) in the votes table. Only votes
// from the last 24 hours count, and older rows are deleted on the next vote.

const VOTE_LIFETIME_MS = 24 * 60 * 60 * 1000;
// The pages re-fetch every 5 seconds, and D1's free plan limits rows read per
// day, so each Worker instance reuses what it last read for a few seconds.
// Writes refresh it, so a voter always sees their own vote right away.
const CACHE_MS = 3000;

export default {
  async fetch(request, env) {
    if (request.method === "OPTIONS") return new Response(null, { status: 204, headers: CORS_HEADERS });
    const url = new URL(request.url);
    if (url.pathname !== "/") return json({ ok: false, error: "Not found." }, 404);

    if (request.method === "GET") {
      if (url.searchParams.get("board") === "resize") return resizeGet(env.DB);
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
    if (body.board === "resize") return resizePost(env.DB, body);
    return votePost(env.DB, body);
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

// ---------- caching ----------

const memory = new Map();

async function cached(key, read) {
  const hit = memory.get(key);
  if (hit && Date.now() - hit.at < CACHE_MS) return hit.value;
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

// ---------- List resize board (Session 3) ----------
//
// GET  ?board=resize -> { ok, submissions: [{ id, type, build, version, points }] }
// POST {"board": "resize", "voter", "type", "build", "version", "points"}
//        -> { ok, id, submissions }
//      points is [[length, bytes], ...], what the student measured with sys.getsizeof.
// POST {"board": "resize", "action": "delete", "voter", "id"} -> { ok, submissions }
//      Removes one of that voter's own submissions.
//
// Submissions are rows of (time, id, voter, type, build, version, points JSON) in
// the resize table. Like votes, they count for 24 hours. Voter ids never leave the
// Worker, so nobody can delete someone else's submission.

const RESIZE_BUILDS = ["append", "literal", "comprehension", "list()", "extend", "repeat", "other"];
const RESIZE_MAX_PER_VOTER = 12;
const RESIZE_MAX_POINTS = 200;

async function resizeGet(db) {
  return json({ ok: true, submissions: await cached("resize", () => readSubmissions(db)) });
}

async function resizePost(db, body) {
  const voter = String(body.voter);
  if (!/^[A-Za-z0-9-]{8,40}$/.test(voter)) return json({ ok: false, error: "Invalid voter." });

  if (body.action === "delete") {
    await db.prepare("DELETE FROM resize WHERE id = ? AND voter = ?").bind(String(body.id), voter).run();
    return json({ ok: true, submissions: await refresh("resize", () => readSubmissions(db)) });
  }

  const type = String(body.type || "").trim();
  const build = String(body.build);
  const version = String(body.version || "").trim();
  const points = body.points;
  if (type.length < 1 || type.length > 30) return json({ ok: false, error: "Describe the element type in 1-30 characters." });
  if (!RESIZE_BUILDS.includes(build)) return json({ ok: false, error: "Unknown build method." });
  if (!/^[0-9]\.[0-9]{1,2}(\.[0-9]{1,2})?[a-z0-9+]{0,8}$/.test(version)) return json({ ok: false, error: "Version should look like 3.12 or 3.12.4." });
  if (!Array.isArray(points) || points.length < 2 || points.length > RESIZE_MAX_POINTS ||
      !points.every((p) => Array.isArray(p) && p.length === 2 &&
        Number.isInteger(p[0]) && p[0] >= 0 && p[0] <= 100000 &&
        Number.isInteger(p[1]) && p[1] >= 0 && p[1] <= 100000000)) {
    return json({ ok: false, error: "Measurements should be 2-200 pairs of whole numbers." });
  }

  // The insert checks the voter's count in the same statement, so two quick
  // submissions can't both slip in under the limit.
  const now = Date.now();
  const cutoff = now - VOTE_LIFETIME_MS;
  const id = crypto.randomUUID().slice(0, 8);
  const [, inserted] = await db.batch([
    db.prepare("DELETE FROM resize WHERE time < ?").bind(cutoff),
    db.prepare(
      `INSERT INTO resize (time, id, voter, type, build, version, points)
       SELECT ?, ?, ?, ?, ?, ?, ?
       WHERE (SELECT COUNT(*) FROM resize WHERE voter = ? AND time >= ?) < ?`,
    ).bind(now, id, voter, type, build, version, JSON.stringify(points), voter, cutoff, RESIZE_MAX_PER_VOTER),
  ]);
  if (inserted.meta.changes === 0) {
    return json({ ok: false, error: "That's the most submissions one person can make. Remove one first." });
  }
  return json({ ok: true, id, submissions: await refresh("resize", () => readSubmissions(db)) });
}

async function readSubmissions(db) {
  const { results } = await db
    .prepare("SELECT id, type, build, version, points FROM resize WHERE time >= ? ORDER BY time")
    .bind(Date.now() - VOTE_LIFETIME_MS)
    .all();
  return results.map((row) => ({ ...row, points: JSON.parse(row.points) }));
}
