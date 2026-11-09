-- Session 7 Dunder Bingo. One row per player per square they've lit; time is when they last lit it.
-- Only the last hour counts, and older rows are deleted on the next write.
CREATE TABLE bingo (
  time INTEGER NOT NULL,
  player TEXT NOT NULL,
  square TEXT NOT NULL,
  PRIMARY KEY (player, square)
);
CREATE INDEX bingo_time ON bingo (time);
