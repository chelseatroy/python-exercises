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

`docs/` holds a one-page poll site for the Session 1 intro examples (`s1_Intro_Examples.ipynb`). Students answer each poll, and only after answering can they open that poll's results. Votes are stored in Firebase Realtime Database, and any vote older than 24 hours is ignored and then deleted, so the page resets itself between quarters.

- Questions and answer choices: `docs/polls.js`
- Database security rules: `database.rules.json`

### One-time setup

1. In the [Firebase console](https://console.firebase.google.com/), create a project (Google Analytics isn't needed). The free Spark plan is enough; it allows 100 simultaneous connections.
2. **Build > Realtime Database > Create database**. Start in locked mode.
3. In the database's **Rules** tab, paste the contents of `database.rules.json` and publish.
4. **Project settings > General > Your apps > Add app > Web**. Copy the `firebaseConfig` values into `docs/firebase-config.js` and commit.
5. On GitHub: **Settings > Pages > Build and deployment**, choose *Deploy from a branch*, branch `main`, folder `/docs`.

The site is served at https://chelseatroy.github.io/python-exercises/.

### How it works

- Each browser gets a random id in `localStorage`. The rules allow one vote per id per poll per 24 hours, and don't allow votes to be edited.
- The page only counts votes from the last 24 hours. When anyone loads the page, it deletes votes older than that; the rules permit deleting a vote only once it has expired.
- Results update live as more votes come in.
