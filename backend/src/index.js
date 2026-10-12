// Backend for the intro polls page in docs/. It's a Cloudflare Worker with a
// D1 database; see the README for setup.
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
