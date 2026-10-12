# Python Exercises

This repository contains Python programming exercises, including a toy tabular data manipulation library called Phoenixcel.

## Setup

### Prerequisites
- Python 3
- pipenv (install with `pip install pipenv`)

### Installation

1. Install dependencies:
```bash
pipenv install --dev
```

This will install:
- **jupyter** - for running Jupyter notebooks
- **pytest** - for running tests

2. Activate the virtual environment:
```bash
pipenv shell
```

## Running Tests

Run all tests:
```bash
pipenv run pytest
```

Run tests with verbose output:
```bash
pipenv run pytest -v
```

Run specific test file:
```bash
pipenv run pytest data_frame_exercise/phoenixcel/tests/test_dataframe.py
```

## Running Jupyter Notebooks

Start Jupyter:
```bash
pipenv run jupyter notebook
```

Or, if you've activated the virtual environment with `pipenv shell`:
```bash
jupyter notebook
```

The notebook server will open in your browser. Navigate to the different directories to access the exercise notebooks for the different exercises.


## Course Website (GitHub Pages)

https://chelseatroy.github.io/python-exercises/ is the course website: the syllabus, a schedule, and an agenda page for each session. GitHub Pages builds it with Jekyll from `docs/` on `main`, so editing a Markdown file and pushing to origin updates the site.

- Syllabus: `docs/syllabus.md`
- Session agendas: `docs/_sessions/s01.md` through `s09.md`. The front matter at the top of each file sets the topic shown on the schedule, the homework time estimate, and (for Session 1) the interactive demos. Each agenda's homework heading ends in `{#homework}`, which the schedule links to.
- Home page and schedule: `docs/index.html` and `docs/schedule.html`, built from the sessions' front matter
- Presentations: `docs/presentations.html`, built from `docs/_data/presentations.yml`. Students add their video links to that data file by pull request.
- Layouts and styles: `docs/_layouts/` and `docs/assets/css/site.css`
- Due-date calendar (on the schedule and in the syllabus): `docs/_includes/due-calendar.html`. It's HTML, not an image, so a date can be changed by moving a box to another week. Each label's arrow points at the box whose `id` matches the label's `data-to`.
- Images: `docs/assets/img/`
- Exercise downloads: `docs/downloads/`, built by `tools/build_downloads.py` (see below)
- Course name, term, and meeting time: `docs/_config.yml`

The interactive demos below are plain HTML pages in `docs/`; Jekyll copies them through unchanged.

### Exercise downloads

Agendas link students to the exact files they need (single files, or a folder as a `.zip`) instead of asking them to clone this repo. Those downloads are copies in `docs/downloads/`, so after changing an exercise that students download, rebuild them and commit the result:

```bash
python3 tools/build_downloads.py          # rewrites anything out of date
python3 tools/build_downloads.py --check  # only reports; exits 1 if something is out of date
```

The list of downloads and where each comes from is `DOWNLOADS` at the top of `tools/build_downloads.py`. Zips include only files tracked by git, without Jupyter checkpoints or `__pycache__`.

## Class Backend (Cloudflare)

The pages that collect answers from students (for now, the intro polls) share one small backend in `backend/`. It's a Cloudflare Worker with a D1 database, on Cloudflare's free plan, served at https://python-exercises-backend.chelseatroy.workers.dev/. The pages read that URL from `docs/s1_polls/config.js`.

- Code: `backend/src/index.js`
- Database tables: `backend/migrations/`. To change a table, add a new numbered `.sql` file rather than editing an old one.
- Config: `backend/wrangler.jsonc`

### Deploying

Pushing a change under `backend/` to `main` on this repo deploys it, through `.github/workflows/deploy-backend.yml`. That workflow applies any new migrations and then deploys the Worker. It uses two repository secrets: `CLOUDFLARE_API_TOKEN` (a Cloudflare API token from the "Edit Cloudflare Workers" template, plus D1 Write) and `CLOUDFLARE_ACCOUNT_ID`. You can also run it by hand from the Actions tab.

To deploy from your own machine instead, run `npx wrangler login` once, then from `backend/`:

```
npx wrangler d1 migrations apply python-exercises-backend --remote
npx wrangler deploy
```

### Trying changes locally

From `backend/`, run `npx wrangler d1 migrations apply python-exercises-backend --local` once, then `npx wrangler dev`. That serves the Worker at http://localhost:8787/ with a local copy of the database. To point the pages at it, change the URL in `docs/s1_polls/config.js` (don't commit that change).

### Looking at or clearing data

Use the D1 console in the Cloudflare dashboard (**Storage & Databases > D1 > python-exercises-backend > Console**), or run SQL from `backend/`:

```
npx wrangler d1 execute python-exercises-backend --remote --command "SELECT * FROM votes"
npx wrangler d1 execute python-exercises-backend --remote --command "DELETE FROM votes"
```

### Free plan limits

The free plan allows 100,000 requests a day, plus 5 million database rows read and 100,000 written. The pages re-fetch every 5 seconds while open, so a full class uses a good share of the request limit in a session. Each Worker instance reuses what it last read from the database for 3 seconds, which keeps rows read well under the limit. Going over a limit makes requests fail until midnight UTC; it never charges you.

## Intro Examples Polls (GitHub Pages)

`docs/s1_polls/` holds a one-page poll site for the Session 1 intro examples (`s1_Intro_Examples.ipynb`). Students answer each poll, and only after answering can they open that poll's results. Votes are stored by the class backend (above). Only votes from the last 24 hours count, so the page resets itself between quarters. It's served at https://chelseatroy.github.io/python-exercises/s1_polls/s1_polls.html.

- Questions and answer choices: `docs/s1_polls/polls.js`
- Backend URL: `docs/s1_polls/config.js`

### How it works

- Each browser gets a random id in `localStorage`. The backend accepts one vote per id per poll per 24 hours. If the same id votes again, it gets back its original answer.
- Votes are rows in the `votes` table. Votes older than 24 hours are left out of the results and deleted on the next vote.
- While a poll's results are open, the page re-fetches the counts every 5 seconds.

`polls-backend.gs` is the earlier Google Sheets version of the backend. It isn't used anymore.

## Binary Envelopes Demo (GitHub Pages)

An interactive page for Session 1 that uses a bank teller's envelopes to show why binary is the most space-efficient way to store numbers in on/off switches. It's served at https://chelseatroy.github.io/python-exercises/s1_binary_envelopes/s1_binary_envelopes.html.

- The page: `docs/s1_binary_envelopes/s1_binary_envelopes.html` (one self-contained file)
- The same logic in Python: `s1_binary_envelopes_demo/envelopes.py` (run it to print a comparison of bases)

## ENIAC Demo (bonus, GitHub Pages)

A bonus page for Session 1 showing how ENIAC stored each decimal digit in a ring of 10 flip-flops, and how many of those switches' patterns go unused compared to binary. It's served at https://chelseatroy.github.io/python-exercises/s1_eniac_demo/s1_eniac_demo.html.

- The page: `docs/s1_eniac_demo/s1_eniac_demo.html` (one self-contained file)

## The Growing List (GitHub Pages)

An interactive page for Session 3 showing how a CPython list stores pointers in a separate block of slots, and how that block grows and shrinks. You can switch between the resize rules from CPython 3.8 and earlier, 3.9 through 3.15.0, the newest change (gh-158592, October 2026), and a hypothetical list with no spare room. It's served at https://chelseatroy.github.io/python-exercises/s3_list_resize/s3_list_resize.html.

- The page: `docs/s3_list_resize/s3_list_resize.html` (one self-contained file)
