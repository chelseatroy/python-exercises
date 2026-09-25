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


## Intro Examples Polls (GitHub Pages)

`docs/` holds a one-page poll site for the Session 1 intro examples (`s1_Intro_Examples.ipynb`). Students answer each poll, and only after answering can they open that poll's results. Votes are stored in a Google Sheet through a small Apps Script web app (`polls-backend.gs`). Only votes from the last 24 hours count, so the page resets itself between quarters.

- Questions and answer choices: `docs/polls.js`
- Backend script: `polls-backend.gs`
- Backend URL: `docs/config.js`

### One-time setup

1. Create a new Google Sheet (any name). Open **Extensions > Apps Script**.
2. Replace the contents of `Code.gs` with `polls-backend.gs` from this repo, and save.
3. Click **Deploy > New deployment**. Under *Select type*, choose **Web app**. Set *Execute as* to **Me** and *Who has access* to **Anyone**. Click **Deploy** and approve the permission prompts. Google warns that the app is unverified because you wrote it yourself: choose *Advanced > Go to (project name)*.
4. Copy the web app URL (it ends in `/exec`) into `docs/config.js` and commit.
5. On GitHub: **Settings > Pages > Build and deployment**, choose *Deploy from a branch*, branch `main`, folder `/docs`.

The site is served at https://chelseatroy.github.io/python-exercises/.

If you change `polls-backend.gs` later, paste it in again and use **Deploy > Manage deployments > Edit > Version: New version**. That keeps the same URL. A *New deployment* gets a new URL.

### How it works

- Each browser gets a random id in `localStorage`. The script accepts one vote per id per poll per 24 hours. If the same id votes again, it gets back its original answer.
- Votes are rows on the sheet's `votes` tab. The first vote creates that tab. Votes older than 24 hours are left out of the results and deleted on the next vote. To clear results by hand, delete every row except the header.
- While a poll's results are open, the page re-fetches the counts every 5 seconds.
