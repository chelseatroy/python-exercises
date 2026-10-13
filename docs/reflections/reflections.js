// Interactive pieces for the Homework 1 reflections page: the approach matrix, the
// empty-file chart, and two small demos. Every mount point is optional, so the page
// still reads fine (with its plain-text table) if this script doesn't load.
(() => {
  "use strict";

  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const COUNT = 10;
  const link = (n) => `<a href="#reflection-${n}">Reflection ${n}</a>`;
  const listOf = (ids) => ids.length ? ids.map(link).join(", ") : "none";

  // ---------- who did what ----------
  // Tagged from what each reflection says. A blank means "not mentioned", not "didn't do it".
  const TECHNIQUES = [
    { group: "Getting started", label: "Dictionary for number words", ids: [1, 2, 3, 4, 8, 10] },
    { group: "Getting started", label: "Dictionary for identifiers", ids: [1, 2, 3, 4, 5] },
    { group: "“The result of”", label: "Turned into parentheses", ids: [5, 6, 8] },
    { group: "“The result of”", label: "Handled by recursion", ids: [2, 3, 4, 6, 9] },
    { group: "Commas", label: "Kept as markers", ids: [3] },
    { group: "Commas", label: "Split the line into chunks", ids: [7, 8, 9] },
    { group: "Order of operations", label: "Two passes: × ÷, then + −", ids: [1, 2, 5, 6, 7] },
    { group: "Upper and lower case", label: "Listed both spellings", ids: [3, 4] },
    { group: "Upper and lower case", label: "Kept two versions of each word", ids: [7] },
    { group: "Habits", label: "Broke it into helper functions", ids: [2, 4, 7, 10] },
    { group: "Habits", label: "Wrote extra test inputs", ids: [1, 2, 6, 9] },
    { group: "Struggles", label: "Index bookkeeping", ids: [2, 4, 5, 6] },
  ];

  const EMPTY_CHOICES = [
    { label: "Write an empty solution file", ids: [1, 3, 5], why: "Consistency: every input gets an output with one line per problem, and zero problems means zero lines. Works well for many files in a row, and needs no special case." },
    { label: "Print a message", ids: [4, 7, 8], why: "Silent success is confusing. Tell people they gave an empty input instead of leaving them a blank file." },
    { label: "Both: a message and an empty file", ids: [2], why: "Consistent output, and the user still finds out." },
    { label: "Raise or report an error", ids: [6], why: "In real software you'd want to hear about suspicious input, instead of blank files piling up." },
    { label: "Unclear, or not done", ids: [9, 10], why: "" },
  ];

  // One selection shared by the matrix and the chart: which reflections to highlight.
  let selection = null; // { kind: "row" | "col" | "empty", index, ids }
  const listeners = [];
  function select(next) {
    const same = selection && next && selection.kind === next.kind && selection.index === next.index;
    selection = same ? null : next;
    for (const fn of listeners) fn();
  }

  // ---------- the matrix ----------
  const matrix = document.getElementById("approach-matrix");
  if (matrix) {
    let html = `<div class="refl-scroll"><table class="refl-matrix"><thead><tr><th scope="col" class="tech-h">Technique</th>`;
    for (let n = 1; n <= COUNT; n++) {
      html += `<th scope="col" class="col-h" data-col="${n}"><button type="button" data-col="${n}" aria-label="Reflection ${n}">${n}</button></th>`;
    }
    html += `<th scope="col" class="count-h">of ${COUNT}</th></tr></thead><tbody>`;
    let group = null;
    TECHNIQUES.forEach((t, i) => {
      if (t.group !== group) {
        group = t.group;
        html += `<tr class="group-row"><th colspan="${COUNT + 2}" scope="colgroup">${esc(group)}</th></tr>`;
      }
      html += `<tr data-row="${i}"><th scope="row"><button type="button" data-row="${i}">${esc(t.label)}</button></th>`;
      for (let n = 1; n <= COUNT; n++) {
        const yes = t.ids.includes(n);
        html += `<td data-col="${n}" class="${yes ? "yes" : ""}">${yes ? `<span class="dot" aria-label="yes"></span>` : `<span class="sr-only">not mentioned</span>`}</td>`;
      }
      html += `<td class="count">${t.ids.length}</td></tr>`;
    });
    html += `</tbody></table></div><p class="refl-picked" aria-live="polite"></p>`;
    matrix.innerHTML = html;

    matrix.addEventListener("click", (e) => {
      const b = e.target.closest("button");
      if (!b) return;
      if (b.dataset.row !== undefined) {
        const i = Number(b.dataset.row);
        select({ kind: "row", index: i, ids: TECHNIQUES[i].ids });
      } else if (b.dataset.col !== undefined) {
        const n = Number(b.dataset.col);
        select({ kind: "col", index: n, ids: [n] });
      }
    });

    const picked = matrix.querySelector(".refl-picked");
    listeners.push(() => {
      const ids = selection ? selection.ids : [];
      for (const el of matrix.querySelectorAll("[data-col]")) {
        el.classList.toggle("on", ids.includes(Number(el.dataset.col)));
      }
      for (const tr of matrix.querySelectorAll("tr[data-row]")) {
        tr.classList.toggle("on", !!selection && selection.kind === "row" && selection.index === Number(tr.dataset.row));
      }
      for (const b of matrix.querySelectorAll("button")) {
        const on = !!selection && ((b.dataset.row !== undefined && selection.kind === "row" && selection.index === Number(b.dataset.row))
          || (b.dataset.col !== undefined && selection.kind === "col" && selection.index === Number(b.dataset.col)));
        b.setAttribute("aria-pressed", String(on));
      }
      if (!selection) {
        picked.innerHTML = "Click a technique to see who mentioned it, or a number to see everything one reflection mentioned.";
      } else if (selection.kind === "row") {
        const t = TECHNIQUES[selection.index];
        picked.innerHTML = `<strong>${esc(t.label)}</strong>: ${t.ids.length} of ${COUNT}. ${listOf(t.ids)}.`;
      } else if (selection.kind === "col") {
        const n = selection.index;
        const used = TECHNIQUES.filter((t) => t.ids.includes(n)).map((t) => esc(t.label.toLowerCase()));
        const empty = EMPTY_CHOICES.find((c) => c.ids.includes(n));
        picked.innerHTML = `${link(n)} mentions: ${used.length ? used.join("; ") : "none of these"}. Empty file: ${esc(empty.label.toLowerCase())}.`;
      } else {
        const c = EMPTY_CHOICES[selection.index];
        picked.innerHTML = `Empty file, <strong>${esc(c.label.toLowerCase())}</strong>: ${listOf(c.ids)}.`;
      }
    });
  }

  // ---------- the empty-file chart ----------
  const chart = document.getElementById("empty-file-chart");
  if (chart) {
    const max = Math.max(...EMPTY_CHOICES.map((c) => c.ids.length));
    chart.innerHTML = `<ul class="refl-bars">${EMPTY_CHOICES.map((c, i) => `
      <li>
        <button type="button" data-choice="${i}" aria-pressed="false">
          <span class="bar-label">${esc(c.label)}</span>
          <span class="bar-track"><span class="bar-fill" style="width:${(c.ids.length / max) * 100}%"></span></span>
          <span class="bar-n">${c.ids.length}</span>
        </button>
        ${c.why ? `<p class="bar-why">${esc(c.why)}</p>` : ""}
      </li>`).join("")}</ul><p class="refl-picked" aria-live="polite"></p>`;
    chart.addEventListener("click", (e) => {
      const b = e.target.closest("button[data-choice]");
      if (!b) return;
      const i = Number(b.dataset.choice);
      select({ kind: "empty", index: i, ids: EMPTY_CHOICES[i].ids });
    });
    const picked = chart.querySelector(".refl-picked");
    listeners.push(() => {
      for (const b of chart.querySelectorAll("button[data-choice]")) {
        b.setAttribute("aria-pressed", String(!!selection && selection.kind === "empty" && selection.index === Number(b.dataset.choice)));
      }
      picked.innerHTML = selection && selection.kind === "empty"
        ? `${listOf(EMPTY_CHOICES[selection.index].ids)}. (They're highlighted in the table at the top, too.)`
        : "Click a bar to see which reflections chose it.";
    });
  }
  for (const fn of listeners) fn();

  // ---------- a tiny calculator, the same logic as the Python examples on the page ----------
  const NUMBERS = { zero: 0, one: 1, two: 2, three: 3, four: 4, five: 5, six: 6, seven: 7, eight: 8, nine: 9, ten: 10,
    eleven: 11, twelve: 12, thirteen: 13, fourteen: 14, fifteen: 15, sixteen: 16, seventeen: 17, eighteen: 18, nineteen: 19, twenty: 20 };
  const OPERATORS = { plus: "+", minus: "-", times: "*", divided: "/" };
  const SHOW = { "+": "+", "-": "−", "*": "×", "/": "÷" };
  const fmt = (x) => Number.isInteger(x) ? String(x) : x.toFixed(2);

  // Two passes: fold × and ÷ into the number before them, then + and − left to right.
  // Parts alternate value, operator, value; each value carries the expression it came from.
  function calculate(parts) {
    if (!parts.length || parts.length % 2 === 0) throw new Error("that doesn't look like number, operator, number…");
    parts.forEach((p, i) => {
      if ((i % 2 === 0) !== (typeof p === "object")) throw new Error("that doesn't look like number, operator, number…");
    });
    const first = [parts[0]];
    for (let i = 1; i < parts.length; i += 2) {
      const op = parts[i], v = parts[i + 1], last = first[first.length - 1];
      if (op === "*" || op === "/") {
        if (op === "/" && v.value === 0) throw new Error("division by zero");
        first[first.length - 1] = { value: op === "*" ? last.value * v.value : last.value / v.value, expr: `${last.expr} ${SHOW[op]} ${v.expr}` };
      } else {
        first.push(op, v);
      }
    }
    let result = first[0];
    for (let i = 1; i < first.length; i += 2) {
      const op = first[i], v = first[i + 1];
      result = { value: op === "+" ? result.value + v.value : result.value - v.value, expr: `${result.expr} ${SHOW[op]} ${v.expr}` };
    }
    return result;
  }

  // ---------- the comma demo ----------
  const comma = document.getElementById("comma-demo");
  if (comma) {
    comma.innerHTML = `
      <label class="demo-label" for="comma-input">Type a problem, or pick one:</label>
      <input id="comma-input" type="text" spellcheck="false" autocomplete="off">
      <div class="demo-presets">
        <button type="button" data-v="six times the result of five plus one plus two">no comma</button>
        <button type="button" data-v="six times the result of five plus one, plus two">with a comma</button>
        <button type="button" data-v="ten minus the result of two plus three, times two">another</button>
      </div>
      <div class="demo-cols">
        <div class="demo-out"><h4>Commas kept as markers</h4><div data-out="keep"></div></div>
        <div class="demo-out"><h4>Commas thrown away</h4><div data-out="drop"></div></div>
      </div>`;
    const input = comma.querySelector("input");

    function tokenize(line) {
      const words = line.toLowerCase().replace(/,/g, " , ").split(/\s+/).filter(Boolean);
      const tokens = [], ignored = [];
      for (let i = 0; i < words.length; i++) {
        const w = words[i];
        if (w === "the" && words[i + 1] === "result" && words[i + 2] === "of") { tokens.push("result_of"); i += 2; }
        else if (w in NUMBERS) tokens.push(NUMBERS[w]);
        else if (w in OPERATORS) tokens.push(OPERATORS[w]);
        else if (w === ",") tokens.push(",");
        else if (w !== "by") ignored.push(w);
      }
      return { tokens, ignored };
    }
    // Same shape as the Python evaluate(tokens, i, nested): returns the value and where to resume.
    function evaluate(tokens, i = 0, nested = false) {
      const parts = [];
      while (i < tokens.length) {
        const t = tokens[i];
        if (t === "result_of") {
          const [inner, next] = evaluate(tokens, i + 1, true);
          parts.push({ value: inner.value, expr: `(${inner.expr})` });
          i = next;
        } else if (t === ",") {
          if (nested) return [calculate(parts), i + 1];
          i += 1;
        } else if (typeof t === "number") {
          parts.push({ value: t, expr: String(t) });
          i += 1;
        } else {
          parts.push(t);
          i += 1;
        }
      }
      return [calculate(parts), i];
    }
    const showTokens = (tokens) => "[" + tokens.map((t) => typeof t === "number" ? t : `'${t}'`).join(", ") + "]";

    function run() {
      const { tokens, ignored } = tokenize(input.value);
      for (const mode of ["keep", "drop"]) {
        const ts = mode === "keep" ? tokens : tokens.filter((t) => t !== ",");
        let body = `<code class="demo-tokens">${esc(showTokens(ts))}</code>`;
        try {
          const [r] = evaluate(ts);
          body += `<p class="demo-expr">${esc(r.expr)}</p><p class="demo-result">= ${fmt(r.value)}</p>`;
        } catch (err) {
          body += `<p class="demo-err">${esc(err.message)}</p>`;
        }
        comma.querySelector(`[data-out="${mode}"]`).innerHTML = body;
      }
      const note = comma.querySelector(".demo-note") || comma.appendChild(Object.assign(document.createElement("p"), { className: "demo-note" }));
      note.textContent = ignored.length ? `Words it doesn't know are skipped: ${ignored.join(", ")}.` : "";
    }
    input.value = "six times the result of five plus one, plus two";
    input.addEventListener("input", run);
    comma.querySelector(".demo-presets").addEventListener("click", (e) => {
      const b = e.target.closest("button");
      if (b) { input.value = b.dataset.v; run(); }
    });
    run();
  }

  // ---------- the case demo ----------
  const caseDemo = document.getElementById("case-demo");
  if (caseDemo) {
    const STRATEGIES = [
      { key: "none", label: "Do nothing" },
      { key: "lower", label: "Lowercase the whole line" },
      { key: "both", label: "List both spellings" },
      { key: "two", label: "Keep two versions of each word" },
    ];
    caseDemo.innerHTML = `
      <label class="demo-label" for="case-input">A short program (edit it if you like; try <code>LET</code> or <code>FIVE</code>):</label>
      <textarea id="case-input" rows="4" spellcheck="false"></textarea>
      <div class="refl-scroll"><table class="case-table"><thead><tr><th scope="col">Line</th>${STRATEGIES.map((s) => `<th scope="col">${s.label}</th>`).join("")}</tr></thead><tbody></tbody></table></div>
      <p class="demo-note">✓ means the line came out the way the assignment asks: keywords and number words in any case, but identifiers case-sensitive, so <code>f</code> and <code>F</code> are different names.</p>`;
    const ta = caseDemo.querySelector("textarea");
    const cap = (w) => w.charAt(0).toUpperCase() + w.slice(1);

    function matcher(strategy) {
      // Does word w count as the keyword or number word k under this strategy?
      if (strategy === "both") return (w, k) => w === k || w === cap(k);
      if (strategy === "two") return (w, k) => w.toLowerCase() === k;
      return (w, k) => w === k;
    }
    function runProgram(lines, strategy) {
      const is = matcher(strategy);
      const lookup = (w, table) => { for (const k in table) if (is(w, k)) return table[k]; return undefined; };
      const env = {};
      return lines.map((raw) => {
        const line = strategy === "lower" ? raw.toLowerCase() : raw;
        const shown = [];
        try {
          for (const clause of line.split(",")) {
            const words = clause.trim().split(/\s+/).filter(Boolean);
            if (!words.length) continue;
            const expr = (ws) => {
              const parts = [];
              for (const w of ws) {
                const n = lookup(w, NUMBERS), op = lookup(w, OPERATORS);
                if (n !== undefined) parts.push({ value: n, expr: String(n) });
                else if (op !== undefined) parts.push(op);
                else if (is(w, "by")) continue;
                else if (w in env) parts.push({ value: env[w], expr: w });
                else throw new Error(`${w} isn't defined`);
              }
              return calculate(parts).value;
            };
            if (is(words[0], "let")) {
              if (words.length < 4 || !is(words[2], "equal")) throw new Error("expected: let NAME equal …");
              env[words[1]] = expr(words.slice(3));
              shown.push(`${words[1]} = ${fmt(env[words[1]])}`);
            } else if (is(words[0], "calculate")) {
              shown.push(fmt(expr(words.slice(1))));
            } else {
              shown.push(fmt(expr(words)));
            }
          }
          return { text: shown.join(" · "), error: false };
        } catch (err) {
          return { text: `error: ${err.message}`, error: true };
        }
      });
    }
    function run() {
      const lines = ta.value.split("\n").filter((l) => l.trim());
      const results = Object.fromEntries(STRATEGIES.map((s) => [s.key, runProgram(lines, s.key)]));
      caseDemo.querySelector("tbody").innerHTML = lines.map((line, i) => {
        const want = results.two[i].text;
        return `<tr><th scope="row"><code>${esc(line)}</code></th>${STRATEGIES.map((s) => {
          const r = results[s.key][i];
          const ok = r.text === want;
          return `<td class="${ok ? "ok" : "bad"}"><span class="mark" aria-label="${ok ? "right" : "wrong"}">${ok ? "✓" : "✗"}</span> ${esc(r.text)}</td>`;
        }).join("")}</tr>`;
      }).join("");
    }
    ta.value = "Let f equal five\nCalculate f plus one\nCalculate F plus one\nLet F equal Two, calculate F times f";
    ta.addEventListener("input", run);
    run();
  }
})();
