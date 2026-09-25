import { POLL_API_URL } from "./config.js";
import { POLLS } from "./polls.js";

// Matches the backend: votes older than this don't count, and this browser
// forgets its own answers after the same span.
const VOTE_LIFETIME_MS = 24 * 60 * 60 * 1000;
// How often to re-fetch counts while any poll's results are open.
const REFRESH_MS = 5000;

const pollList = document.getElementById("polls");
const banner = document.getElementById("banner");

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
      // Storage blocked (e.g. private mode): the value lasts until the page reloads.
      try { localStorage.setItem(key, JSON.stringify(value)); } catch {}
    },
  };
}

function getVoterId() {
  const store = storage("intro-polls-voter-id", () => null);
  let id = store.load();
  if (typeof id !== "string") {
    id = crypto.randomUUID();
    store.save(id);
  }
  return id;
}

// This browser's own answers, as { pollId: { c: choiceIndex, t: timeAnswered } }.
const answerStore = storage("intro-polls-answers", () => ({}));

function loadAnswers() {
  const cutoff = Date.now() - VOTE_LIFETIME_MS;
  const answers = answerStore.load();
  return Object.fromEntries(Object.entries(answers || {}).filter(([, a]) => a && a.t >= cutoff));
}

function saveAnswer(pollId, choice) {
  const answers = loadAnswers();
  answers[pollId] = { c: choice, t: Date.now() };
  answerStore.save(answers);
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

async function callApi(options) {
  const response = await fetch(POLL_API_URL, options);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const data = await response.json();
  if (!data.ok) throw new Error(data.error || "Request failed");
  return data;
}

// A plain-string body goes out as text/plain, which skips the CORS preflight
// request that Apps Script can't answer.
function submitVote(pollId, choice) {
  return callApi({ method: "POST", body: JSON.stringify({ poll: pollId, voter: voterId, choice }) });
}

function fetchCounts() {
  return callApi({ cache: "no-store" });
}

function renderPoll(poll, index) {
  const state = { counts: null, myChoice: null, showingResults: false, submitting: false };
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
    if (!state.counts) {
      results.replaceChildren(el("h3", { text: "Loading results…" }));
      return;
    }
    const counts = poll.options.map((_, i) => state.counts[i] || 0);
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
      const data = await submitVote(poll.id, Number(picked.value));
      state.myChoice = data.choice;
      saveAnswer(poll.id, data.choice);
      setAllCounts(data.counts);
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
    if (state.showingResults) refreshCounts();
    scheduleRefresh();
  });

  return {
    card,
    get showingResults() { return state.showingResults; },
    setMyChoice(choice) {
      state.myChoice = choice;
      update();
    },
    setCounts(counts) {
      state.counts = counts || [];
      update();
    },
    disable() {
      for (const label of radios) label.querySelector("input").disabled = true;
      submitButton.disabled = true;
    },
  };
}

function setAllCounts(counts) {
  POLLS.forEach((poll, i) => polls[i].setCounts(counts[poll.id]));
}

let refreshing = false;
async function refreshCounts() {
  if (refreshing) return;
  refreshing = true;
  try {
    setAllCounts((await fetchCounts()).counts);
    banner.hidden = true;
  } catch (error) {
    console.error(error);
    showBanner("Couldn't load results. They'll retry in a few seconds.");
  }
  refreshing = false;
}

// Re-fetch counts every few seconds, but only while someone is looking at results.
let refreshTimer = null;
function scheduleRefresh() {
  const wanted = configured && !document.hidden && polls.some((p) => p.showingResults);
  if (wanted && refreshTimer === null) {
    refreshTimer = setInterval(refreshCounts, REFRESH_MS);
  } else if (!wanted && refreshTimer !== null) {
    clearInterval(refreshTimer);
    refreshTimer = null;
  }
}

const voterId = getVoterId();
const configured = !POLL_API_URL.includes("REPLACE_ME");

const polls = POLLS.map(renderPoll);
pollList.append(...polls.map((p) => p.card));

if (!configured) {
  showBanner("Voting isn't set up yet: docs/config.js still has a placeholder URL.");
  polls.forEach((p) => p.disable());
} else {
  const answers = loadAnswers();
  POLLS.forEach((poll, i) => {
    if (answers[poll.id]) polls[i].setMyChoice(answers[poll.id].c);
  });
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden && polls.some((p) => p.showingResults)) refreshCounts();
    scheduleRefresh();
  });
}
