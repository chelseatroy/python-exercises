-- Poll votes. One vote per voter per poll; rows older than 24 hours are deleted on the next vote.
CREATE TABLE votes (
  time INTEGER NOT NULL,
  poll TEXT NOT NULL,
  voter TEXT NOT NULL,
  choice INTEGER NOT NULL,
  PRIMARY KEY (poll, voter)
);
CREATE INDEX votes_time ON votes (time);

-- Session 3 list resize board submissions. points is a JSON array of [length, bytes] pairs.
CREATE TABLE resize (
  time INTEGER NOT NULL,
  id TEXT PRIMARY KEY,
  voter TEXT NOT NULL,
  type TEXT NOT NULL,
  build TEXT NOT NULL,
  version TEXT NOT NULL,
  points TEXT NOT NULL
);
CREATE INDEX resize_time ON resize (time);
CREATE INDEX resize_voter ON resize (voter);
