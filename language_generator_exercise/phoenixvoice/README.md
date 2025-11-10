# PhoenixVoice

A Python library for generating text in the style of any WordPress blog author.

## Installation

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

**Required packages:**
- For **Markov models**: `requests`, `beautifulsoup4`


## Quick Start

### Fetch Blog Posts

First, download blog posts from any WordPress site using the fetch_blog_posts function.

```bash
blog_file = fetch_blog_posts(base_url="https://chelseatroy.com", max_posts=450)
```

This creates a file like `chelseatroy_com_blog_posts.txt` containing all the blog post text.

### Generate Text

Use the unified CLI to generate text in the author's voice:

```bash
# Train on blog posts
markov_generator = MarkovVoiceGenerator(order=3)
markov_generator.train(blog_file)

# Generate text
markov_text = markov_generator.generate_text(max_tokens=300)

# Save for later
markov_generator.save("my_model.pkl")

```

**Tuning parameters:**
- `--order`: Higher order (2-3) = more coherent but requires more data
- `--max-tokens`: Length of generated text


## Running Tests

```bash
# Run all tests
python -m unittest discover -s test -p "test_*.py"

# Run specific test suite
python -m unittest test.test_markov_voice_generator -v
python -m unittest test.test_factory -v
```