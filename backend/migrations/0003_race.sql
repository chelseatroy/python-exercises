-- Session 8 Lost Updates. One row of shared state: the round, its mode, when it ends,
-- the shared counter, and (in lock mode) who holds the lock and until when.
CREATE TABLE race_state (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  round INTEGER NOT NULL,
  mode TEXT NOT NULL,
  ends INTEGER NOT NULL,
  counter INTEGER NOT NULL,
  holder TEXT,
  holder_until INTEGER NOT NULL
);
INSERT INTO race_state (id, round, mode, ends, counter, holder, holder_until) VALUES (1, 0, 'race', 0, 0, NULL, 0);

-- One row per tap that reached the counter: the value the player read, the value they
-- wrote (read + 1), and what the counter held just before their write. A tap erased
-- before - read other taps. Rows older than three hours are deleted on the next round.
CREATE TABLE race_taps (
  round INTEGER NOT NULL,
  mode TEXT NOT NULL,
  time INTEGER NOT NULL,
  player TEXT NOT NULL,
  read INTEGER NOT NULL,
  before INTEGER NOT NULL
);
CREATE INDEX race_taps_round ON race_taps (round);
