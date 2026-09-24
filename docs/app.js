import { initializeApp } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-app.js";
import {
  getDatabase, ref, query, orderByChild, startAt, endAt,
  onValue, get, set, update, serverTimestamp,
} from "https://www.gstatic.com/firebasejs/10.12.2/firebase-database.js";
import { firebaseConfig } from "./firebase-config.js";
import { POLLS } from "./polls.js";

// Votes older than this are ignored, and deleted by whichever browser loads the page next.
const VOTE_LIFETIME_MS = 24 * 60 * 60 * 1000;
// Slack for clock differences between this browser and the database server,
// whose rules only permit deleting votes that are a full day old.
const CLEANUP_MARGIN_MS = 10 * 60 * 1000;

const pollList = document.getElementById("polls");
const banner = document.getElementById("banner");

function getVoterId() {
  const key = "intro-polls-voter-id";
  try {
    let id = localStorage.getItem(key);
    if (!id) {
      id = crypto.randomUUID();
      localStorage.setItem(key, id);
    }
    return id;
  } catch {
    // Storage blocked (e.g. private mode): the id lasts until the page reloads.
    return crypto.randomUUID();
  }
}

function showBanner(message) {
  banner.textContent = message;
  banner.hidden = false;
}

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [name, value] of Object.entries(attrs)) {
    if (name === "class") node.className = value;
    else if (name === "text") node.textContent = value;
    else node.setAttribute(name, value);
  }
  node.append(...children);
  return node;
}

function recentVotes(votes) {
  const cutoff = Date.now() - VOTE_LIFETIME_MS;
  return Object.entries(votes || {}).filter(([, vote]) => vote && vote.t >= cutoff);
}

function renderPoll(poll, index) {
  const state = { votes: {}, myChoice: null, showingResults: false, submitting: false };
  const name = `poll-${poll.id}`;

  const radios = poll.options.map((option, i) =>
    el("label", { class: "option" },
      el("input", { type: "radio", name, value: String(i) }),
      el("code", { text: option })));
  const fieldset = el("fieldset", {},
    el("legend", { class: "question", text: poll.question }),
    ...radios);

  const submitButton = el("button", { type: "submit", class: "primary", text: "Submit answer" });
  const resultsButton = el("button", { type: "button", text: "See results", disabled: "" });
  const status = el("p", { class: "status", "aria-live": "polite" });
  const results = el("div", { class: "results", "aria-live": "polite", hidden: "" });

  const form = el("form", {}, fieldset, el("div", { class: "actions" }, submitButton, resultsButton), status);

  const card = el("article", { class: "poll", id: poll.id },
    el("p", { class: "eyebrow", text: `Poll ${index + 1} · ${poll.scenario}` }),
    el("pre", {}, el("code", { text: poll.code })),
    form,
    results);

  function update() {
    const answered = state.myChoice !== null;
    for (const label of radios) {
      const input = label.querySelector("input");
      input.disabled = answered || state.submitting;
      if (answered) input.checked = Number(input.value) === state.myChoice;
    }
    submitButton.hidden = answered;
    submitButton.disabled = state.submitting;
    resultsButton.disabled = !answered;
    resultsButton.textContent = state.showingResults ? "Hide results" : "See results";
    if (answered) status.textContent = "Your answer is in.";
    results.hidden = !(answered && state.showingResults);
    if (!results.hidden) drawResults();
  }

  function drawResults() {
    const counts = poll.options.map(() => 0);
    const votes = recentVotes(state.votes);
    for (const [, vote] of votes) {
      if (Number.isInteger(vote.c) && vote.c < counts.length) counts[vote.c] += 1;
    }
    const total = counts.reduce((a, b) => a + b, 0);
    const max = Math.max(1, ...counts);

    const rows = poll.options.map((option, i) => {
      const pct = total ? Math.round((counts[i] / total) * 100) : 0;
      const bar = el("div", { class: "bar" });
      bar.style.width = `${(counts[i] / max) * 100}%`;
      return el("li", { class: "row", title: `${option}: ${counts[i]} of ${total} (${pct}%)` },
        el("div", { class: "row-label" },
          el("code", { text: option }),
          i === state.myChoice ? el("span", { class: "mine", text: "your answer" }) : ""),
        el("div", { class: "track" }, bar),
        el("span", { class: "value", text: `${counts[i]} · ${pct}%` }));
    });

    results.replaceChildren(
      el("h3", { text: `Results: ${total} ${total === 1 ? "response" : "responses"} in the last 24 hours` }),
      el("ol", { class: "chart" }, ...rows));
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const picked = form.querySelector(`input[name="${name}"]:checked`);
    if (!picked) {
      status.textContent = "Pick an answer first.";
      return;
    }
    state.submitting = true;
    status.textContent = "Submitting…";
    update();
    try {
      await set(ref(db, `votes/${poll.id}/${voterId}`), { c: Number(picked.value), t: serverTimestamp() });
      state.myChoice = Number(picked.value);
    } catch (error) {
      console.error(error);
      status.textContent = "That didn't go through. Check your connection and try again.";
    }
    state.submitting = false;
    update();
  });

  resultsButton.addEventListener("click", () => {
    state.showingResults = !state.showingResults;
    update();
  });

  return {
    card,
    setVotes(votes) {
      state.votes = votes || {};
      const mine = recentVotes(state.votes).find(([voter]) => voter === voterId);
      if (mine) state.myChoice = mine[1].c;
      update();
    },
    disable() {
      for (const label of radios) label.querySelector("input").disabled = true;
      submitButton.disabled = true;
    },
  };
}

async function deleteExpiredVotes(pollId) {
  const pollRef = ref(db, `votes/${pollId}`);
  const expired = await get(query(pollRef, orderByChild("t"),
    endAt(Date.now() - VOTE_LIFETIME_MS - CLEANUP_MARGIN_MS)));
  if (!expired.exists()) return;
  const deletions = {};
  expired.forEach((child) => { deletions[child.key] = null; });
  await update(pollRef, deletions);
}

const voterId = getVoterId();
const configured = !Object.values(firebaseConfig).some((value) => value.includes("REPLACE_ME"));
const db = configured ? getDatabase(initializeApp(firebaseConfig)) : null;

const polls = POLLS.map(renderPoll);
pollList.append(...polls.map((p) => p.card));

if (!db) {
  showBanner("Voting isn't set up yet: docs/firebase-config.js still has placeholder values.");
  polls.forEach((p) => p.disable());
} else {
  POLLS.forEach((poll, i) => {
    const recent = query(ref(db, `votes/${poll.id}`), orderByChild("t"),
      startAt(Date.now() - VOTE_LIFETIME_MS));
    onValue(recent, (snapshot) => polls[i].setVotes(snapshot.val()), (error) => {
      console.error(error);
      showBanner("Couldn't load votes. Try reloading the page.");
    });
    deleteExpiredVotes(poll.id).catch((error) => console.warn("Cleanup skipped:", error));
  });
}
