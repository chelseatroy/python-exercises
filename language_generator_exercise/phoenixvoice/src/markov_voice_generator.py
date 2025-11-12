"""
Markov model voice generator.

This implements probabilistic text generation using Markov chains,
where the next token is chosen based on the frequency of transitions
observed in the training data.

Good for learning about:
- Probabilistic text generation fundamentals
- N-gram models and state transitions
- Simple statistical language modeling
- Trade-offs between model order and data sparsity
"""

import re
import random
import pickle
from collections import defaultdict
from .voice_generator import VoiceGenerator


class MarkovModel:
    def __init__(self, order=1):
        """
        Initialize a Markov model.

        Args:
            order: The order of the Markov model (how many previous tokens to consider)
                   1 = first-order (look at 1 previous token)
                   2 = second-order (look at 2 previous tokens)
                   etc.
        """
        self.order = order
        self.transitions = {}
        self.start_states = []
        self.total_transitions = {}

    def add_post(self, post_text):
        """
        Add a blog post to the Markov model.

        Args:
            post_text: The full text of a blog post
        """
        tokens = self._tokenize(post_text)

        if not tokens:
            return

        # For higher order models, we need at least order+1 tokens
        if len(tokens) <= self.order:
            return

        # Track start states for generation
        start_state = tuple(tokens[:self.order])
        self.start_states.append(start_state)

        # Build transition counts
        for i in range(len(tokens) - self.order):
            # Current state is a tuple of 'order' tokens
            current_state = tuple(tokens[i:i + self.order])
            next_token = tokens[i + self.order]

            if current_state not in self.transitions:
                self.transitions[current_state] = {}
            if next_token not in self.transitions[current_state]:
                self.transitions[current_state][next_token] = 0

            self.transitions[current_state][next_token] += 1

            if current_state not in self.total_transitions:
                self.total_transitions[current_state] = 0
            self.total_transitions[current_state] += 1

    def _tokenize(self, text):
        """
        Tokenize text into words and punctuation.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        # Split on whitespace and punctuation, but keep punctuation as tokens
        # Also preserve special tokens like <START> and <END>
        tokens = re.findall(r'<\w+>|\w+|[.,!?;:]', text)
        return tokens

    def get_transition_probability(self, current_state, next_token):
        """
        Get the probability of transitioning from current_state to next_token.

        Args:
            current_state: The current state (tuple of tokens)
            next_token: The next token

        Returns:
            Probability as a float between 0 and 1
        """
        if current_state not in self.transitions:
            return 0.0

        count = self.transitions[current_state].get(next_token, 0)
        total = self.total_transitions[current_state]

        return count / total if total > 0 else 0.0

    def get_next_token_probabilities(self, current_state):
        """
        Get all possible next tokens and their probabilities.

        Args:
            current_state: The current state (tuple of tokens)

        Returns:
            Dictionary mapping next tokens to their probabilities
        """
        if current_state not in self.transitions:
            return {}

        total = self.total_transitions[current_state]
        probabilities = {}

        for next_token, count in self.transitions[current_state].items():
            probabilities[next_token] = count / total

        return probabilities

    def generate_text(self, max_tokens=100, start_state=None):
        """
        Generate text using the Markov model.

        Args:
            max_tokens: Maximum number of tokens to generate
            start_state: Optional starting state (tuple of tokens). If None, picks random start

        Returns:
            Generated text as a string
        """
        if not self.start_states:
            return ""

        if start_state is None:
            current_state = random.choice(self.start_states)
        else:
            current_state = start_state

        # Start with the tokens from the initial state
        generated = list(current_state)

        for _ in range(max_tokens - self.order):
            if current_state not in self.transitions:
                break

            # Get next token probabilities
            next_tokens = list(self.transitions[current_state].keys())
            weights = [self.transitions[current_state][t] for t in next_tokens]

            # Choose next token based on weights
            next_token = random.choices(next_tokens, weights=weights)[0]
            generated.append(next_token)

            # Update state: drop first token, add new token at end
            current_state = tuple(list(current_state[1:]) + [next_token])

        # Join tokens into text, handling punctuation spacing
        text = self._detokenize(generated)
        return text

    def _detokenize(self, tokens):
        """
        Convert tokens back to readable text.

        Args:
            tokens: List of tokens

        Returns:
            Formatted text string
        """
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
        """
        Save the trained Markov model to a file.

        Args:
            filename: Path to save the model to (will be pickled)
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {filename}")

    @staticmethod
    def load(filename):
        """
        Load a trained Markov model from a file.

        Args:
            filename: Path to the saved model file

        Returns:
            MarkovModel instance
        """
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {filename}")
        return model

    def __len__(self):
        """
        Return the number of unique states in the model.

        This shows how much the model has learned. More states generally
        means the model has seen more diverse text patterns.

        Returns:
            Number of unique states (int)

        Example:
            model = MarkovModel(order=2)
            model.add_post("The quick brown fox")
            print(len(model))  # Shows number of 2-token states
        """
        return len(self.transitions)

    def __getitem__(self, state):
        """
        Get transition probabilities for a given state.

        This allows accessing the model like a dictionary to see what tokens
        can follow a given state and with what probability.

        Args:
            state: Either a tuple of tokens (e.g., ("the", "quick")) or
                   a single string (e.g., "the") which will be converted
                   to a 1-tuple for convenience

        Returns:
            Dictionary mapping next tokens to their probabilities

        Raises:
            KeyError: If the state doesn't exist in the model

        Example:
            model = MarkovModel(order=2)
            model.add_post("The quick brown fox jumps")
            probs = model[("the", "quick")]
            # Returns: {"brown": 1.0}

            # For 1st-order models, you can use a string:
            model1 = MarkovModel(order=1)
            model1.add_post("The quick brown fox")
            probs = model1["quick"]  # Automatically converts to ("quick",)
        """
        # Convert string to tuple for convenience
        if isinstance(state, str):
            state = (state,)

        # Check if state exists
        if state not in self.transitions:
            raise KeyError(f"State {state} not found in model")

        return self.get_next_token_probabilities(state)


def parse_blog_posts_file(filename):
    """
    Parse blog posts from the fetched blog posts file.

    Args:
        filename: Path to the blog posts text file

    Returns:
        List of post texts (content after "Reading Time: X minutes")
    """
    posts = []

    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by post separators (80 equal signs)
    post_sections = content.split('=' * 80)

    for section in post_sections:
        section = section.strip()
        if not section:
            continue

        # Look for "Reading Time:" and extract everything after it
        match = re.search(r'Reading Time:.*?minutes?\s+(.*)', section, re.DOTALL | re.IGNORECASE)

        if match:
            post_text = match.group(1).strip()
            if post_text:
                posts.append(post_text)
        else:
            # If no "Reading Time:" found, just take the content after the header
            lines = section.split('\n')
            # Skip first few lines (POST N:, Date:, URL:)
            content_start = 0
            for i, line in enumerate(lines):
                if line.startswith('POST ') or line.startswith('Date:') or line.startswith('URL:'):
                    content_start = i + 1

            if content_start < len(lines):
                post_text = '\n'.join(lines[content_start:]).strip()
                if post_text:
                    posts.append(post_text)

    return posts


def build_markov_model_from_file(filename, order=1):
    """
    Build a Markov model from a blog posts file.

    Args:
        filename: Path to the blog posts text file
        order: The order of the Markov model (default: 1)

    Returns:
        MarkovModel instance trained on the posts
    """
    posts = parse_blog_posts_file(filename)

    model = MarkovModel(order=order)

    for post in posts:
        # Add special start/end tokens
        post_with_markers = f"<START> {post} <END>"
        model.add_post(post_with_markers)

    print(f"Built {order}-order Markov model from {len(posts)} posts")
    print(f"Total unique states: {len(model.transitions)}")

    return model


class MarkovVoiceGenerator(VoiceGenerator):
    """
    VoiceGenerator implementation using Markov models.

    This wrapper adapts the MarkovModel class to conform to the VoiceGenerator interface,
    allowing easy switching between Markov and transformer-based generation.
    """

    def __init__(self, order=1):
        """
        Initialize a Markov-based voice generator.

        Args:
            order: The order of the Markov model (default: 1)
        """
        super().__init__()
        self.order = order
        self.model = None

    def train(self, text_file_path):
        """
        Train the Markov model on text from a file.

        Args:
            text_file_path: Path to text file containing blog posts
        """
        self.model = build_markov_model_from_file(text_file_path, order=self.order)
        self._is_trained = True

    def generate_text(self, max_tokens=50, num_samples=1):
        """
        Generate text samples using the trained Markov model.

        Args:
            max_tokens: Maximum number of tokens to generate
            num_samples: Number of text samples to generate

        Returns:
            If num_samples=1: string of generated text
            If num_samples>1: list of generated text strings
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before generating text. Call train() first.")

        samples = []
        for _ in range(num_samples):
            text = self.model.generate_text(max_tokens=max_tokens)
            samples.append(text)

        return samples[0] if num_samples == 1 else samples

    def save(self, filepath):
        """
        Save the trained Markov model to disk.

        Args:
            filepath: Path where model should be saved
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before saving. Call train() first.")

        self.model.save(filepath)

    def load(self, filepath):
        """
        Load a previously trained Markov model from disk.

        Args:
            filepath: Path to saved model file
        """
        self.model = MarkovModel.load(filepath)
        self.order = self.model.order
        self._is_trained = True
