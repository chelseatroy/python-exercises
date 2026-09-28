"""
Simple RNN/LSTM voice generator.

This implements a basic recurrent neural network for text generation,
providing an introduction to neural sequence modeling without the complexity
of full transformer architectures.

Good for learning about:
- Recurrent neural networks: Processing sequences with hidden state
- LSTM cells: Handling long-range dependencies
- Backpropagation through time: Training sequential models
- Character vs. word-level modeling
- Temperature sampling: Controlling randomness

Installation required:
    pip install torch
"""

import re
import random
import pickle
import os
from .voice_generator import VoiceGenerator

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


class TextDataset(Dataset):
    """Dataset for character-level text generation."""

    def __init__(self, text, seq_length=50):
        """
        Args:
            text: Full text as a string
            seq_length: Length of sequences for training
        """
        self.text = text
        self.seq_length = seq_length

        # Build character vocabulary
        self.chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

    def __len__(self):
        return len(self.text) - self.seq_length

    def __getitem__(self, idx):
        # Get input sequence and target (next character)
        input_seq = self.text[idx:idx + self.seq_length]
        target = self.text[idx + self.seq_length]

        # Convert to indices
        input_indices = [self.char_to_idx[ch] for ch in input_seq]
        target_index = self.char_to_idx[target]

        return torch.tensor(input_indices), torch.tensor(target_index)


class CharRNN(nn.Module):
    """
    Character-level RNN model.

    This is a simple LSTM-based model for generating text one character at a time.
    """

    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2):
        """
        Args:
            vocab_size: Number of unique characters
            embedding_dim: Dimension of character embeddings
            hidden_dim: Dimension of LSTM hidden state
            num_layers: Number of LSTM layers
        """
        super(CharRNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Embedding layer: maps character indices to dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM layer: processes sequences
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )

        # Output layer: maps hidden state to character probabilities
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_length)
            hidden: Optional hidden state tuple (h, c)

        Returns:
            Output logits and new hidden state
        """
        # Embed characters
        embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)

        # Pass through LSTM
        if hidden is None:
            lstm_out, hidden = self.lstm(embedded)
        else:
            lstm_out, hidden = self.lstm(embedded, hidden)

        # Get predictions for each time step
        output = self.fc(lstm_out)  # (batch, seq_len, vocab_size)

        return output, hidden

    def init_hidden(self, batch_size, device):
        """Initialize hidden state with zeros."""
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h, c)


class RNNModel:
    """
    RNN-based text generation model.

    This trains a character-level LSTM and generates text by sampling
    from the output distribution.
    """

    def __init__(
        self,
        embedding_dim=128,
        hidden_dim=256,
        num_layers=2,
        seq_length=50,
        batch_size=64,
        learning_rate=0.002,
        num_epochs=10,
        temperature=0.8
    ):
        """
        Initialize RNN model.

        Args:
            embedding_dim: Dimension of character embeddings (default: 128)
            hidden_dim: Dimension of LSTM hidden state (default: 256)
            num_layers: Number of LSTM layers (default: 2)
            seq_length: Training sequence length (default: 50)
            batch_size: Training batch size (default: 64)
            learning_rate: Learning rate (default: 0.002)
            num_epochs: Number of training epochs (default: 10)
            temperature: Sampling temperature (default: 0.8)
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for RNN generation. "
                "Install with: pip install torch"
            )

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.temperature = temperature

        self.model = None
        self.dataset = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self, text):
        """
        Train the RNN model on text.

        Args:
            text: Full text as a string
        """
        print(f"Training character-level RNN on {len(text)} characters...")
        print(f"Using device: {self.device}")

        # Create dataset
        self.dataset = TextDataset(text, seq_length=self.seq_length)
        print(f"Vocabulary size: {self.dataset.vocab_size}")

        # Create model
        self.model = CharRNN(
            vocab_size=self.dataset.vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers
        ).to(self.device)

        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Create data loader
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True
        )

        # Training loop
        self.model.train()
        for epoch in range(self.num_epochs):
            total_loss = 0
            num_batches = 0

            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                optimizer.zero_grad()
                output, _ = self.model(inputs)

                # Calculate loss (only on last character of sequence)
                loss = criterion(output[:, -1, :], targets)

                # Backward pass
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                # Print progress occasionally
                if batch_idx % 100 == 0:
                    print(f"Epoch {epoch+1}/{self.num_epochs}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")

            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch+1}/{self.num_epochs} completed. Average loss: {avg_loss:.4f}")

        print("Training completed!")

    def generate_text(self, max_chars=500, seed=None):
        """
        Generate text using the trained RNN.

        Args:
            max_chars: Maximum number of characters to generate
            seed: Optional seed string to start generation

        Returns:
            Generated text as a string
        """
        if self.model is None:
            raise RuntimeError("Model must be trained before generating text.")

        self.model.eval()

        with torch.no_grad():
            # Initialize with seed or random character
            if seed:
                current_text = seed
            else:
                current_text = random.choice(self.dataset.chars)

            # Ensure seed is long enough
            while len(current_text) < self.seq_length:
                current_text = random.choice(self.dataset.chars) + current_text

            generated = current_text

            # Generate characters
            for _ in range(max_chars):
                # Prepare input (last seq_length characters)
                input_seq = generated[-self.seq_length:]
                input_indices = [self.dataset.char_to_idx[ch] for ch in input_seq]
                input_tensor = torch.tensor([input_indices]).to(self.device)

                # Get prediction
                output, _ = self.model(input_tensor)

                # Apply temperature sampling
                logits = output[0, -1, :] / self.temperature
                probs = torch.softmax(logits, dim=0)

                # Sample next character
                next_idx = torch.multinomial(probs, 1).item()
                next_char = self.dataset.idx_to_char[next_idx]

                generated += next_char

            # Return only the newly generated part (exclude seed)
            return generated[len(current_text):]

    def save(self, filepath):
        """Save the trained RNN model to disk."""
        if self.model is None:
            raise RuntimeError("Model must be trained before saving.")

        # Save model state
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'char_to_idx': self.dataset.char_to_idx,
            'idx_to_char': self.dataset.idx_to_char,
            'chars': self.dataset.chars,
            'vocab_size': self.dataset.vocab_size,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'seq_length': self.seq_length,
            'temperature': self.temperature
        }, filepath)

        print(f"RNN model saved to {filepath}")

    @staticmethod
    def load(filepath):
        """Load a trained RNN model from disk."""
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for RNN generation. "
                "Install with: pip install torch"
            )

        # Load checkpoint
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(filepath, map_location=device)

        # Create model instance
        model = RNNModel(
            embedding_dim=checkpoint['embedding_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            num_layers=checkpoint['num_layers'],
            seq_length=checkpoint['seq_length'],
            temperature=checkpoint['temperature']
        )

        # Restore model
        model.model = CharRNN(
            vocab_size=checkpoint['vocab_size'],
            embedding_dim=checkpoint['embedding_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            num_layers=checkpoint['num_layers']
        ).to(device)

        model.model.load_state_dict(checkpoint['model_state_dict'])

        # Restore dataset info
        class DummyDataset:
            pass

        model.dataset = DummyDataset()
        model.dataset.char_to_idx = checkpoint['char_to_idx']
        model.dataset.idx_to_char = checkpoint['idx_to_char']
        model.dataset.chars = checkpoint['chars']
        model.dataset.vocab_size = checkpoint['vocab_size']

        print(f"RNN model loaded from {filepath}")
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


def build_rnn_model_from_file(filename, **kwargs):
    """
    Build an RNN model from a blog posts file.

    Args:
        filename: Path to the blog posts text file
        **kwargs: Arguments to pass to RNNModel constructor

    Returns:
        Trained RNNModel instance
    """
    posts = parse_blog_posts_file(filename)
    text = '\n\n'.join(posts)

    print(f"Building RNN model from {len(posts)} posts")
    print(f"Total characters: {len(text)}")

    model = RNNModel(**kwargs)
    model.train(text)

    return model


class RNNVoiceGenerator(VoiceGenerator):
    """
    VoiceGenerator implementation using RNN/LSTM.

    This is a neural approach that:
    - Learns character-level patterns (not just word sequences)
    - Can generate novel character combinations
    - Captures style through learned representations
    - Requires more compute but can be more flexible

    Pedagogical value:
    - Introduction to neural sequence modeling
    - Understanding of recurrent architectures
    - Experience with training neural networks
    - Comparison between statistical and neural approaches
    """

    def __init__(
        self,
        embedding_dim=128,
        hidden_dim=256,
        num_layers=2,
        seq_length=50,
        batch_size=64,
        learning_rate=0.002,
        num_epochs=10,
        temperature=0.8
    ):
        """
        Initialize an RNN voice generator.

        Args:
            embedding_dim: Dimension of character embeddings (default: 128)
            hidden_dim: Dimension of LSTM hidden state (default: 256)
            num_layers: Number of LSTM layers (default: 2)
            seq_length: Training sequence length (default: 50)
            batch_size: Training batch size (default: 64)
            learning_rate: Learning rate (default: 0.002)
            num_epochs: Number of training epochs (default: 10)
            temperature: Sampling temperature (default: 0.8)
        """
        super().__init__()

        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for RNN generation. "
                "Install with: pip install torch"
            )

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.temperature = temperature
        self.model = None

    def train(self, text_file_path):
        """Train the RNN model on text from a file."""
        self.model = build_rnn_model_from_file(
            text_file_path,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            seq_length=self.seq_length,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            num_epochs=self.num_epochs,
            temperature=self.temperature
        )
        self._is_trained = True

    def generate_text(self, max_tokens=50, num_samples=1, seed=None):
        """
        Generate text samples using the trained RNN model.

        Note: max_tokens is interpreted as max_chars for character-level models.

        Args:
            max_tokens: Maximum number of characters to generate (not words)
            num_samples: Number of text samples to generate
            seed: Optional seed string to start generation

        Returns:
            If num_samples=1: string of generated text
            If num_samples>1: list of generated text strings
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before generating text. Call train() first.")

        # For character-level model, interpret max_tokens as max_chars
        # Multiply by 5 to roughly convert words to characters
        max_chars = max_tokens * 5 if max_tokens < 200 else max_tokens

        samples = []
        for _ in range(num_samples):
            text = self.model.generate_text(max_chars=max_chars, seed=seed)
            samples.append(text)

        return samples[0] if num_samples == 1 else samples

    def save(self, filepath):
        """Save the trained RNN model to disk."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before saving. Call train() first.")

        self.model.save(filepath)

    def load(self, filepath):
        """Load a previously trained RNN model from disk."""
        self.model = RNNModel.load(filepath)
        self.embedding_dim = self.model.embedding_dim
        self.hidden_dim = self.model.hidden_dim
        self.num_layers = self.model.num_layers
        self.seq_length = self.model.seq_length
        self.temperature = self.model.temperature
        self._is_trained = True
