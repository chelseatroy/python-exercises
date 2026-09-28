"""
Word2Vec-based voice generator.

This approach uses word embeddings to capture semantic relationships between words,
then generates text by sampling words that are semantically similar to the context.

Good for learning about:
- Word embeddings: Representing words as dense vectors
- Semantic similarity: Words with similar meanings have similar vectors
- Vector operations: Arithmetic in embedding space
- Distributed representations: Meaning encoded across dimensions

Installation required:
    pip install gensim
"""

import re
import random
import pickle
import os
from .voice_generator import VoiceGenerator

try:
    from gensim.models import Word2Vec
    import numpy as np
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    Word2Vec = None
    np = None


class Word2VecModel:
    """
    Word2Vec-based text generation model.

    This model learns word embeddings and generates text by:
    1. Starting with a seed phrase
    2. Finding words semantically similar to recent context
    3. Sampling from similar words with temperature-based randomness
    """

    def __init__(self, vector_size=100, window=5, min_count=2, temperature=0.7):
        """
        Initialize a Word2Vec generation model.

        Args:
            vector_size: Dimensionality of word vectors (default: 100)
            window: Context window size for training (default: 5)
            min_count: Ignore words with frequency below this (default: 2)
            temperature: Sampling temperature (higher = more random) (default: 0.7)
        """
        if not GENSIM_AVAILABLE:
            raise ImportError(
                "gensim is required for Word2Vec generation. "
                "Install with: pip install gensim"
            )

        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.temperature = temperature
        self.model = None
        self.sentences = []
        self.vocabulary = set()

    def add_post(self, post_text):
        """
        Add a blog post to the training data.

        Args:
            post_text: The full text of a blog post
        """
        tokens = self._tokenize(post_text)

        if not tokens:
            return

        self.sentences.append(tokens)
        self.vocabulary.update(tokens)

    def train(self):
        """
        Train the Word2Vec model on the collected sentences.
        """
        if not self.sentences:
            raise ValueError("No training data. Add posts first with add_post().")

        print(f"Training Word2Vec on {len(self.sentences)} sentences...")
        print(f"Vocabulary size: {len(self.vocabulary)}")

        self.model = Word2Vec(
            sentences=self.sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=4,
            epochs=10
        )

        print(f"Word2Vec model trained with {len(self.model.wv)} word vectors")

    def _tokenize(self, text):
        """Tokenize text into words (excluding punctuation for embeddings)."""
        # For Word2Vec, we typically only use words, not punctuation
        tokens = re.findall(r'<\w+>|\w+', text.lower())
        return tokens

    def generate_text(self, max_tokens=100, seed=None):
        """
        Generate text using Word2Vec embeddings.

        Strategy:
        1. Start with a seed phrase (or random words from vocabulary)
        2. For each step, find words similar to recent context
        3. Sample from similar words using temperature
        4. Add punctuation heuristically

        Args:
            max_tokens: Maximum number of tokens to generate
            seed: Optional seed phrase (string or list of tokens)

        Returns:
            Generated text as a string
        """
        if self.model is None:
            raise RuntimeError("Model must be trained before generating text.")

        # Initialize with seed or random words
        if seed:
            if isinstance(seed, str):
                current_tokens = self._tokenize(seed)
            else:
                current_tokens = seed
        else:
            # Start with random words from vocabulary
            vocab_list = list(self.model.wv.index_to_key)
            current_tokens = [random.choice(vocab_list) for _ in range(3)]

        generated = list(current_tokens)

        for _ in range(max_tokens - len(current_tokens)):
            try:
                # Get context (last few tokens that are in vocabulary)
                context = [t for t in generated[-5:] if t in self.model.wv]

                if not context:
                    # Fall back to random word
                    next_token = random.choice(list(self.model.wv.index_to_key))
                else:
                    # Find words similar to the context
                    # Average the context vectors
                    context_vector = np.mean([self.model.wv[word] for word in context], axis=0)

                    # Find similar words
                    similar_words = self.model.wv.similar_by_vector(
                        context_vector,
                        topn=20
                    )

                    # Sample from similar words using temperature
                    words = [word for word, _ in similar_words]
                    similarities = [sim for _, sim in similar_words]

                    # Apply temperature to similarities
                    # Higher temperature = more uniform, lower = more peaked
                    temps = np.array(similarities) / self.temperature
                    probs = np.exp(temps) / np.sum(np.exp(temps))

                    next_token = np.random.choice(words, p=probs)

                generated.append(next_token)

            except (KeyError, ValueError):
                # If we get an error, just pick a random word
                next_token = random.choice(list(self.model.wv.index_to_key))
                generated.append(next_token)

        # Add punctuation heuristically
        text = self._add_punctuation(generated)
        return text

    def _add_punctuation(self, tokens):
        """
        Add punctuation to a sequence of word tokens.

        This is a simple heuristic approach:
        - Capitalize first word
        - Add periods every 10-15 words
        - Add commas occasionally
        """
        if not tokens:
            return ""

        result = []
        tokens_since_period = 0

        for i, token in enumerate(tokens):
            # Capitalize first word or word after period
            if i == 0 or (result and result[-1] == '.'):
                token = token.capitalize()

            result.append(token)
            tokens_since_period += 1

            # Add comma occasionally (10% chance)
            if random.random() < 0.1 and tokens_since_period > 3:
                result.append(',')

            # Add period every 10-15 words
            if tokens_since_period > random.randint(10, 15):
                result.append('.')
                tokens_since_period = 0

        # Ensure we end with a period
        if result and result[-1] not in '.!?':
            result.append('.')

        # Join tokens, handling punctuation spacing
        text = ""
        for i, token in enumerate(result):
            if token in '.,!?;:':
                text += token
            elif i == 0:
                text += token
            else:
                text += " " + token

        return text

    def get_similar_words(self, word, topn=10):
        """
        Get words similar to a given word.

        Useful for exploring the learned embeddings.

        Args:
            word: Word to find similar words for
            topn: Number of similar words to return

        Returns:
            List of (word, similarity) tuples
        """
        if self.model is None:
            raise RuntimeError("Model must be trained first.")

        if word not in self.model.wv:
            raise KeyError(f"Word '{word}' not in vocabulary")

        return self.model.wv.most_similar(word, topn=topn)

    def save(self, filepath):
        """Save the Word2Vec model to disk."""
        if self.model is None:
            raise RuntimeError("Model must be trained before saving.")

        # Save Word2Vec model
        model_path = filepath + '.w2v'
        self.model.save(model_path)

        # Save metadata
        metadata = {
            'vector_size': self.vector_size,
            'window': self.window,
            'min_count': self.min_count,
            'temperature': self.temperature,
            'vocabulary': self.vocabulary
        }

        with open(filepath + '.meta', 'wb') as f:
            pickle.dump(metadata, f)

        print(f"Word2Vec model saved to {filepath}")

    @staticmethod
    def load(filepath):
        """Load a Word2Vec model from disk."""
        if not GENSIM_AVAILABLE:
            raise ImportError(
                "gensim is required for Word2Vec generation. "
                "Install with: pip install gensim"
            )

        # Load metadata
        with open(filepath + '.meta', 'rb') as f:
            metadata = pickle.load(f)

        # Create model instance
        model = Word2VecModel(
            vector_size=metadata['vector_size'],
            window=metadata['window'],
            min_count=metadata['min_count'],
            temperature=metadata['temperature']
        )

        # Load Word2Vec model
        model.model = Word2Vec.load(filepath + '.w2v')
        model.vocabulary = metadata['vocabulary']

        print(f"Word2Vec model loaded from {filepath}")
        return model


def parse_blog_posts_file(filename):
    """Parse blog posts from the fetched blog posts file."""
    posts = []

    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    post_sections = content.split('=' * 80)

    for section in post_sections:
        section = section.strip()
        if not section:
            continue

        match = re.search(r'Reading Time:.*?minutes?\s+(.*)', section, re.DOTALL | re.IGNORECASE)

        if match:
            post_text = match.group(1).strip()
            if post_text:
                posts.append(post_text)
        else:
            lines = section.split('\n')
            content_start = 0
            for i, line in enumerate(lines):
                if line.startswith('POST ') or line.startswith('Date:') or line.startswith('URL:'):
                    content_start = i + 1

            if content_start < len(lines):
                post_text = '\n'.join(lines[content_start:]).strip()
                if post_text:
                    posts.append(post_text)

    return posts


def build_word2vec_model_from_file(filename, vector_size=100, window=5, min_count=2, temperature=0.7):
    """
    Build a Word2Vec model from a blog posts file.

    Args:
        filename: Path to the blog posts text file
        vector_size: Dimensionality of word vectors (default: 100)
        window: Context window size (default: 5)
        min_count: Minimum word frequency (default: 2)
        temperature: Sampling temperature (default: 0.7)

    Returns:
        Word2VecModel instance trained on the posts
    """
    posts = parse_blog_posts_file(filename)

    model = Word2VecModel(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        temperature=temperature
    )

    for post in posts:
        model.add_post(post)

    model.train()

    return model


class Word2VecVoiceGenerator(VoiceGenerator):
    """
    VoiceGenerator implementation using Word2Vec embeddings.

    This approach is fundamentally different from Markov/n-gram models:
    - Uses semantic similarity rather than sequential patterns
    - Can generate more diverse text (not just seen sequences)
    - May be less coherent but more creative

    Pedagogical value:
    - Introduces vector representations of words
    - Shows how semantics can be captured numerically
    - Demonstrates sampling with temperature
    - Illustrates the trade-off between coherence and creativity
    """

    def __init__(self, vector_size=100, window=5, min_count=2, temperature=0.7):
        """
        Initialize a Word2Vec voice generator.

        Args:
            vector_size: Dimensionality of word vectors (default: 100)
            window: Context window size (default: 5)
            min_count: Minimum word frequency (default: 2)
            temperature: Sampling temperature (default: 0.7)
        """
        super().__init__()

        if not GENSIM_AVAILABLE:
            raise ImportError(
                "gensim is required for Word2Vec generation. "
                "Install with: pip install gensim"
            )

        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.temperature = temperature
        self.model = None

    def train(self, text_file_path):
        """Train the Word2Vec model on text from a file."""
        self.model = build_word2vec_model_from_file(
            text_file_path,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            temperature=self.temperature
        )
        self._is_trained = True

    def generate_text(self, max_tokens=50, num_samples=1, seed=None):
        """
        Generate text samples using the trained Word2Vec model.

        Args:
            max_tokens: Maximum number of tokens to generate
            num_samples: Number of text samples to generate
            seed: Optional seed phrase to start generation

        Returns:
            If num_samples=1: string of generated text
            If num_samples>1: list of generated text strings
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before generating text. Call train() first.")

        samples = []
        for _ in range(num_samples):
            text = self.model.generate_text(max_tokens=max_tokens, seed=seed)
            samples.append(text)

        return samples[0] if num_samples == 1 else samples

    def get_similar_words(self, word, topn=10):
        """
        Get words similar to a given word.

        This is useful for exploring what the model has learned.

        Args:
            word: Word to find similar words for
            topn: Number of similar words to return

        Returns:
            List of (word, similarity) tuples
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained first. Call train() first.")

        return self.model.get_similar_words(word, topn=topn)

    def save(self, filepath):
        """Save the trained Word2Vec model to disk."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before saving. Call train() first.")

        self.model.save(filepath)

    def load(self, filepath):
        """Load a previously trained Word2Vec model from disk."""
        self.model = Word2VecModel.load(filepath)
        self.vector_size = self.model.vector_size
        self.window = self.model.window
        self.min_count = self.model.min_count
        self.temperature = self.model.temperature
        self._is_trained = True
