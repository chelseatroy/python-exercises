"""
N-gram with backoff voice generator.

This implements a more sophisticated version of the Markov model that handles
sparse data better by "backing off" to lower-order n-grams when higher-order
ones aren't available.

Good for learning about:
- Smoothing: Handling unseen n-grams gracefully
- Backoff: Using lower-order models as fallback
- Probability interpolation: Combining different order models
"""

import re
import random
import pickle
from collections import defaultdict
from .voice_generator import VoiceGenerator


class NgramModel:
    """
    N-gram model with backoff support.

    Unlike a pure Markov model, this can fall back to lower-order n-grams
    when higher-order ones aren't found, making it more robust to sparse data.
    """

    def __init__(self, max_order=3, smoothing_alpha=0.1):
        """
        Initialize an n-gram model with backoff.

        Args:
            max_order: Maximum n-gram order to use (e.g., 3 for trigrams)
            smoothing_alpha: Smoothing parameter for unseen n-grams (default: 0.1)
        """
        self.max_order = max_order
        self.smoothing_alpha = smoothing_alpha

        # Store n-grams of all orders from 1 to max_order
        # models[n] contains n-grams of order n
        self.models = {}
        for order in range(1, max_order + 1):
            self.models[order] = {
                'transitions': {},
                'total_transitions': {},
                'start_states': []
            }

        self.vocabulary = set()

    def add_post(self, post_text):
        """
        Add a blog post to all n-gram models.

        Args:
            post_text: The full text of a blog post
        """
        tokens = self._tokenize(post_text)

        if not tokens:
            return

        # Add to vocabulary
        self.vocabulary.update(tokens)

        # Build n-grams of all orders
        for order in range(1, self.max_order + 1):
            if len(tokens) <= order:
                continue

            # Track start states
            start_state = tuple(tokens[:order])
            self.models[order]['start_states'].append(start_state)

            # Build transitions
            for i in range(len(tokens) - order):
                current_state = tuple(tokens[i:i + order])
                next_token = tokens[i + order]

                transitions = self.models[order]['transitions']
                total_transitions = self.models[order]['total_transitions']

                if current_state not in transitions:
                    transitions[current_state] = {}
                if next_token not in transitions[current_state]:
                    transitions[current_state][next_token] = 0

                transitions[current_state][next_token] += 1

                if current_state not in total_transitions:
                    total_transitions[current_state] = 0
                total_transitions[current_state] += 1

    def _tokenize(self, text):
        """Tokenize text into words and punctuation."""
        tokens = re.findall(r'<\w+>|\w+|[.,!?;:]', text)
        return tokens

    def get_probability_with_backoff(self, context, next_token):
        """
        Get probability of next_token given context, using backoff.

        This tries to use the highest-order model first, but backs off to
        lower orders if the context isn't found.

        Args:
            context: Tuple of tokens representing the context
            next_token: The token to predict

        Returns:
            Probability as a float
        """
        order = len(context)

        # Try current order
        if order in self.models:
            transitions = self.models[order]['transitions']
            total_transitions = self.models[order]['total_transitions']

            if context in transitions:
                count = transitions[context].get(next_token, 0)
                total = total_transitions[context]

                if count > 0:
                    # Apply smoothing
                    vocab_size = len(self.vocabulary)
                    smoothed_prob = (count + self.smoothing_alpha) / (total + self.smoothing_alpha * vocab_size)
                    return smoothed_prob

        # Backoff to lower order
        if order > 1:
            # Use the suffix of the context (drop first token)
            return self.get_probability_with_backoff(context[1:], next_token)

        # Base case: uniform distribution over vocabulary
        return 1.0 / len(self.vocabulary) if self.vocabulary else 0.0

    def get_next_token_probabilities_with_backoff(self, context):
        """
        Get all possible next tokens and their probabilities using backoff.

        Args:
            context: Tuple of tokens representing the context

        Returns:
            Dictionary mapping next tokens to their probabilities
        """
        order = len(context)
        probabilities = {}

        # Try current order first
        if order in self.models and context in self.models[order]['transitions']:
            transitions = self.models[order]['transitions'][context]
            total = self.models[order]['total_transitions'][context]
            vocab_size = len(self.vocabulary)

            for token, count in transitions.items():
                # Smoothed probability
                probabilities[token] = (count + self.smoothing_alpha) / (total + self.smoothing_alpha * vocab_size)

            return probabilities

        # Backoff to lower order
        if order > 1:
            return self.get_next_token_probabilities_with_backoff(context[1:])

        # Base case: uniform distribution
        return {token: 1.0 / len(self.vocabulary) for token in self.vocabulary}

    def generate_text(self, max_tokens=100, start_state=None):
        """
        Generate text using n-gram model with backoff.

        Args:
            max_tokens: Maximum number of tokens to generate
            start_state: Optional starting state (tuple of tokens)

        Returns:
            Generated text as a string
        """
        if not self.models[self.max_order]['start_states']:
            return ""

        if start_state is None:
            current_state = random.choice(self.models[self.max_order]['start_states'])
        else:
            current_state = start_state

        generated = list(current_state)

        for _ in range(max_tokens - self.max_order):
            # Get probabilities using backoff
            probs = self.get_next_token_probabilities_with_backoff(current_state)

            if not probs:
                break

            # Sample next token
            tokens = list(probs.keys())
            weights = list(probs.values())
            next_token = random.choices(tokens, weights=weights)[0]

            generated.append(next_token)

            # Update state
            current_state = tuple(list(current_state[1:]) + [next_token])

        return self._detokenize(generated)

    def _detokenize(self, tokens):
        """Convert tokens back to readable text."""
        text = ""
        for i, token in enumerate(tokens):
            if token in '.,!?;:':
                text += token
            elif i == 0:
                text += token
            else:
                text += " " + token
        return text

    def save(self, filename):
        """Save the n-gram model to a file."""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        print(f"N-gram model saved to {filename}")

    @staticmethod
    def load(filename):
        """Load an n-gram model from a file."""
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        print(f"N-gram model loaded from {filename}")
        return model

    def __len__(self):
        """Return total number of unique n-grams across all orders."""
        return sum(len(model['transitions']) for model in self.models.values())


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


def build_ngram_model_from_file(filename, max_order=3, smoothing_alpha=0.1):
    """
    Build an n-gram model with backoff from a blog posts file.

    Args:
        filename: Path to the blog posts text file
        max_order: Maximum n-gram order (default: 3)
        smoothing_alpha: Smoothing parameter (default: 0.1)

    Returns:
        NgramModel instance trained on the posts
    """
    posts = parse_blog_posts_file(filename)

    model = NgramModel(max_order=max_order, smoothing_alpha=smoothing_alpha)

    for post in posts:
        post_with_markers = f"<START> {post} <END>"
        model.add_post(post_with_markers)

    print(f"Built {max_order}-order n-gram model from {len(posts)} posts")
    print(f"Total unique n-grams: {len(model)}")
    print(f"Vocabulary size: {len(model.vocabulary)}")

    return model


class NgramVoiceGenerator(VoiceGenerator):
    """
    VoiceGenerator implementation using n-gram models with backoff.

    This is more sophisticated than pure Markov models because it:
    1. Handles unseen n-grams gracefully through backoff
    2. Uses smoothing to assign non-zero probability to rare events
    3. Maintains multiple order models simultaneously

    Pedagogical value:
    - Shows how to handle sparse data in probabilistic models
    - Demonstrates the concept of model interpolation
    - Introduces smoothing techniques
    """

    def __init__(self, max_order=3, smoothing_alpha=0.1):
        """
        Initialize an n-gram voice generator with backoff.

        Args:
            max_order: Maximum n-gram order (default: 3)
            smoothing_alpha: Smoothing parameter for unseen n-grams (default: 0.1)
        """
        super().__init__()
        self.max_order = max_order
        self.smoothing_alpha = smoothing_alpha
        self.model = None

    def train(self, text_file_path):
        """Train the n-gram model on text from a file."""
        self.model = build_ngram_model_from_file(
            text_file_path,
            max_order=self.max_order,
            smoothing_alpha=self.smoothing_alpha
        )
        self._is_trained = True

    def generate_text(self, max_tokens=50, num_samples=1):
        """Generate text samples using the trained n-gram model."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before generating text. Call train() first.")

        samples = []
        for _ in range(num_samples):
            text = self.model.generate_text(max_tokens=max_tokens)
            samples.append(text)

        return samples[0] if num_samples == 1 else samples

    def save(self, filepath):
        """Save the trained n-gram model to disk."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before saving. Call train() first.")

        self.model.save(filepath)

    def load(self, filepath):
        """Load a previously trained n-gram model from disk."""
        self.model = NgramModel.load(filepath)
        self.max_order = self.model.max_order
        self.smoothing_alpha = self.model.smoothing_alpha
        self._is_trained = True
