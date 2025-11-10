# PhoenixVoice

A Python library for generating text in the style of any WordPress blog author using either Markov chains or fine-tuned transformer models.

## Installation

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

**Required packages:**
- For **Markov models only**: `requests`, `beautifulsoup4`
- For **Transformer models** (in addition to above): `torch`, `transformers`, `datasets`, `accelerate`

**Note:** The transformer dependencies are large (~2-3GB total). The `accelerate` package is **required** to train transformer models - without it, you'll get an ImportError during training. If you only want to use Markov models, you can install just the minimal dependencies:

```bash
pip install requests beautifulsoup4
```

## Quick Start

### Fetch Blog Posts

First, download blog posts from any WordPress site:

```bash
python src/fetch_blog_posts.py --url https://chelseatroy.com --max-posts 450
```

This creates a file like `chelseatroy_com_blog_posts.txt` containing all the blog post text.

### Generate Text

Use the unified CLI to generate text in the author's voice:

```bash
# Using Markov model (fast, lightweight)
python src/generate_voice.py --model-type markov --order 2 --train --text-file chelseatroy_com_blog_posts.txt --samples 3

# Using Transformer model (slower, higher quality)
python src/generate_voice.py --model-type transformer --train --text-file chelseatroy_com_blog_posts.txt --samples 3
```

## Two Methods for Voice Generation

PhoenixVoice supports two different approaches for learning and mimicking a writing voice. Both use the same interface, making it easy to switch between them.

### Method 1: Markov Chains

**How it works:** Builds a statistical model based on token sequences. Given the last N tokens, it predicts the next token based on frequency in the training data.

**Pros:**
- Fast training (seconds)
- Small model size (~10-40MB)
- No GPU required
- Predictable behavior

**Cons:**
- Output can be choppy or nonsensical
- Limited coherence over long sequences
- Can't learn complex patterns

**Best for:** Quick experiments, small datasets, low-resource environments

**Example:**
```bash
# Train a 2nd-order Markov model (looks at last 2 tokens)
python src/generate_voice.py \
    --model-type markov \
    --order 2 \
    --train \
    --text-file chelseatroy_com_blog_posts.txt \
    --samples 5 \
    --max-tokens 100 \
    --save my_markov_model.pkl

# Load and use saved model
python src/generate_voice.py \
    --model-type markov \
    --load my_markov_model.pkl \
    --samples 3
```

**Tuning parameters:**
- `--order`: Higher order (2-3) = more coherent but requires more data
- `--max-tokens`: Length of generated text

### Method 2: Transformer (DistilGPT-2)

**How it works:** Fine-tunes a pre-trained language model (DistilGPT-2, 82M parameters) on the blog posts. The model learns deep patterns in writing style, vocabulary, and structure.

**Pros:**
- Higher quality, more coherent output
- Better at capturing writing style
- Can generate longer coherent passages
- Leverages pre-trained language understanding

**Cons:**
- Slow training (minutes to hours)
- Large model size (~165MB+)
- Benefits from GPU (works on CPU but slow)
- Requires more memory

**Best for:** Production use, high-quality generation, larger datasets

**Example:**
```bash
# Train a transformer model
python src/generate_voice.py \
    --model-type transformer \
    --train \
    --text-file chelseatroy_com_blog_posts.txt \
    --samples 3 \
    --max-tokens 100 \
    --save my_transformer_model \
    --num-epochs 3 \
    --learning-rate 5e-5

# Load and use saved model
python src/generate_voice.py \
    --model-type transformer \
    --load my_transformer_model \
    --samples 3 \
    --temperature 0.8
```

**Tuning parameters:**
- `--temperature`: Lower (0.6-0.8) = more focused, higher (1.0-1.5) = more creative
- `--num-epochs`: More epochs = better fit to style (but risk overfitting)
- `--learning-rate`: Default 5e-5 usually works well
- `--batch-size`: Adjust based on available memory

## Programmatic Usage

You can also use PhoenixVoice as a library:

```python
from phoenixvoice import fetch_posts, create_voice_generator

# Fetch blog posts from any WordPress site
blog_file = fetch_posts(base_url="https://chelseatroy.com", max_posts=50)

# Create a generator (markov or transformer)
generator = create_voice_generator("markov", order=2)

# Train on blog posts
generator.train(blog_file)

# Generate text
text = generator.generate_text(max_tokens=100, num_samples=1)
print(text)

# Save for later
generator.save("my_model.pkl")

# Load a saved model
new_generator = create_voice_generator("markov")
new_generator.load("my_model.pkl")
```

**Switching between methods is trivial:**
```python
# Just change this one line:
generator = create_voice_generator("transformer", temperature=0.8)
# Everything else stays the same!
```

## Comparison Table

| Feature | Markov | Transformer |
|---------|--------|-------------|
| Training time | Seconds | Minutes to hours |
| Model size | 10-40MB | 165MB+ |
| GPU required | No | Recommended |
| Memory usage | Low (~100MB) | High (2-8GB) |
| Output quality | Medium | High |
| Coherence | Short-range | Long-range |
| Setup complexity | Simple | Moderate |

## Running Tests

```bash
# Run all tests
python -m unittest discover -s test -p "test_*.py"

# Run specific test suite
python -m unittest test.test_markov_voice_generator -v
python -m unittest test.test_factory -v
```

## Common Issues

**Q: Error: "Using the `Trainer` with `PyTorch` requires `accelerate>=0.21.0`"**
- This means the `accelerate` package is not installed
- Fix: `pip install accelerate>=0.21.0`
- Or reinstall all dependencies: `pip install -r requirements.txt`
- The accelerate package is required for transformer training

**Q: Transformer training is very slow**
- Use a GPU if available
- Reduce `--batch-size` (e.g., 2 or 1)
- Reduce `--num-epochs`
- Consider using Markov instead for quick iterations

**Q: Out of memory errors with transformer**
- Reduce `--batch-size`
- Reduce `--max-length` parameter
- Close other applications
- Use Markov model instead

**Q: Generated text doesn't sound like the author**
- For Markov: Increase `--order` (try 2 or 3)
- For Transformer: Increase `--num-epochs`, ensure you have enough training data (200+ posts recommended)
- For both: Make sure you're training on clean, representative text

**Q: How much training data do I need?**
- Markov: Works with 10+ posts, better with 50+
- Transformer: Minimum 50 posts, 200+ recommended for best results

## License

This is an educational project for learning about text generation techniques.
