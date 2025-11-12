"""
Unit tests for RNN Voice Generator.

Tests the RNN/LSTM-based text generation functionality.
Note: These tests require PyTorch to be installed and may take longer to run.
"""

import unittest
import tempfile
import os

# Check if PyTorch is available
try:
    from phoenixvoice.src.rnn_voice_generator import (
        CharRNN,
        RNNModel,
        RNNVoiceGenerator,
        build_rnn_model_from_file,
        TORCH_AVAILABLE
    )
    TESTS_CAN_RUN = TORCH_AVAILABLE
except ImportError:
    TESTS_CAN_RUN = False
    CharRNN = None
    RNNModel = None
    RNNVoiceGenerator = None


@unittest.skipIf(not TESTS_CAN_RUN, "PyTorch not available")
class TestCharRNN(unittest.TestCase):
    """Test the CharRNN model class."""

    def setUp(self):
        """Set up test fixtures."""
        import torch
        self.vocab_size = 50
        self.embedding_dim = 32
        self.hidden_dim = 64
        self.num_layers = 2
        self.device = torch.device('cpu')

        self.model = CharRNN(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers
        ).to(self.device)

    def test_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.model.hidden_dim, self.hidden_dim)
        self.assertEqual(self.model.num_layers, self.num_layers)

    def test_forward_pass(self):
        """Test forward pass through the model."""
        import torch

        batch_size = 4
        seq_length = 10

        # Create dummy input
        x = torch.randint(0, self.vocab_size, (batch_size, seq_length)).to(self.device)

        # Forward pass
        output, hidden = self.model(x)

        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_length, self.vocab_size))

        # Check hidden state shape
        h, c = hidden
        self.assertEqual(h.shape, (self.num_layers, batch_size, self.hidden_dim))
        self.assertEqual(c.shape, (self.num_layers, batch_size, self.hidden_dim))

    def test_forward_with_hidden_state(self):
        """Test forward pass with provided hidden state."""
        import torch

        batch_size = 4
        seq_length = 10

        x = torch.randint(0, self.vocab_size, (batch_size, seq_length)).to(self.device)

        # Initialize hidden state
        hidden = self.model.init_hidden(batch_size, self.device)

        # Forward pass with hidden state
        output, new_hidden = self.model(x, hidden)

        self.assertEqual(output.shape, (batch_size, seq_length, self.vocab_size))


@unittest.skipIf(not TESTS_CAN_RUN, "PyTorch not available")
class TestRNNModel(unittest.TestCase):
    """Test the RNNModel class."""

    def setUp(self):
        """Set up test fixtures."""
        # Use small parameters for fast testing
        self.model = RNNModel(
            embedding_dim=32,
            hidden_dim=64,
            num_layers=1,
            seq_length=20,
            batch_size=4,
            learning_rate=0.01,
            num_epochs=1,  # Just 1 epoch for testing
            temperature=0.8
        )

    def test_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.model.embedding_dim, 32)
        self.assertEqual(self.model.hidden_dim, 64)
        self.assertEqual(self.model.num_layers, 1)
        self.assertIsNone(self.model.model)

    def test_train_on_text(self):
        """Test training on a text corpus."""
        # Create sample text (repeated for sufficient training data)
        text = "The quick brown fox jumps over the lazy dog. " * 100

        # Train model (just 1 epoch for speed)
        self.model.train(text)

        # Check model was created
        self.assertIsNotNone(self.model.model)
        self.assertIsNotNone(self.model.dataset)

    def test_generate_text(self):
        """Test text generation."""
        # Train on sample text
        text = "Hello world this is a test. " * 100
        self.model.train(text)

        # Generate text
        generated = self.model.generate_text(max_chars=50)

        self.assertIsInstance(generated, str)
        self.assertGreater(len(generated), 0)
        self.assertLessEqual(len(generated), 50)

    def test_generate_with_seed(self):
        """Test generation with seed text."""
        text = "Hello world this is a test. " * 100
        self.model.train(text)

        generated = self.model.generate_text(max_chars=50, seed="Hello")

        self.assertIsInstance(generated, str)
        self.assertGreater(len(generated), 0)

    def test_save_and_load(self):
        """Test saving and loading the model."""
        # Train model
        text = "Sample text for testing. " * 100
        self.model.train(text)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.pth') as f:
            temp_path = f.name

        try:
            self.model.save(temp_path)

            # Load model
            loaded_model = RNNModel.load(temp_path)

            # Verify parameters
            self.assertEqual(loaded_model.embedding_dim, self.model.embedding_dim)
            self.assertEqual(loaded_model.hidden_dim, self.model.hidden_dim)

            # Test generation works
            generated = loaded_model.generate_text(max_chars=30)
            self.assertIsInstance(generated, str)

        finally:
            os.unlink(temp_path)


@unittest.skipIf(not TESTS_CAN_RUN, "PyTorch not available")
class TestRNNVoiceGenerator(unittest.TestCase):
    """Test the RNNVoiceGenerator class."""

    def setUp(self):
        """Set up test fixtures."""
        # Use minimal parameters for fast testing
        self.generator = RNNVoiceGenerator(
            embedding_dim=32,
            hidden_dim=64,
            num_layers=1,
            seq_length=20,
            batch_size=4,
            learning_rate=0.01,
            num_epochs=1,
            temperature=0.8
        )

        # Create a temporary training file
        self.temp_file = tempfile.NamedTemporaryFile(
            mode='w',
            delete=False,
            suffix='.txt',
            encoding='utf-8'
        )

        # Write sample blog posts (repeated for sufficient data)
        for i in range(3):
            self.temp_file.write('=' * 80 + '\n')
            self.temp_file.write(f'POST {i+1}: Sample Post\n')
            self.temp_file.write('Date: 2024-01-01\n')
            self.temp_file.write('Reading Time: 5 minutes\n\n')
            self.temp_file.write(
                'This is sample text for training the RNN model. '
                'It needs to have enough characters to train properly. '
            ) * 20
            self.temp_file.write('\n\n')

        self.temp_file.close()

    def tearDown(self):
        """Clean up test fixtures."""
        os.unlink(self.temp_file.name)

    def test_initialization(self):
        """Test generator initialization."""
        self.assertEqual(self.generator.embedding_dim, 32)
        self.assertEqual(self.generator.hidden_dim, 64)
        self.assertFalse(self.generator.is_trained)

    def test_train(self):
        """Test training the generator."""
        self.generator.train(self.temp_file.name)

        self.assertTrue(self.generator.is_trained)
        self.assertIsNotNone(self.generator.model)

    def test_generate_text_before_training(self):
        """Test that generating before training raises an error."""
        with self.assertRaises(RuntimeError):
            self.generator.generate_text()

    def test_generate_single_sample(self):
        """Test generating a single text sample."""
        self.generator.train(self.temp_file.name)

        # max_tokens is interpreted as words, multiplied by ~5 for chars
        text = self.generator.generate_text(max_tokens=30, num_samples=1)

        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)

    def test_generate_multiple_samples(self):
        """Test generating multiple text samples."""
        self.generator.train(self.temp_file.name)

        texts = self.generator.generate_text(max_tokens=30, num_samples=2)

        self.assertIsInstance(texts, list)
        self.assertEqual(len(texts), 2)
        for text in texts:
            self.assertIsInstance(text, str)
            self.assertGreater(len(text), 0)

    def test_generate_with_seed(self):
        """Test generation with seed text."""
        self.generator.train(self.temp_file.name)

        text = self.generator.generate_text(max_tokens=30, seed="This is")

        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)

    def test_save_and_load(self):
        """Test saving and loading the generator."""
        self.generator.train(self.temp_file.name)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.pth') as f:
            save_path = f.name

        try:
            self.generator.save(save_path)

            # Create new generator and load
            new_generator = RNNVoiceGenerator()
            new_generator.load(save_path)

            # Verify
            self.assertTrue(new_generator.is_trained)
            self.assertEqual(new_generator.embedding_dim, self.generator.embedding_dim)

            # Test generation works
            text = new_generator.generate_text(max_tokens=20)
            self.assertIsInstance(text, str)

        finally:
            os.unlink(save_path)

    def test_string_representation(self):
        """Test __str__ and __repr__ methods."""
        str_repr = str(self.generator)
        self.assertIn("untrained", str_repr.lower())

        self.generator.train(self.temp_file.name)

        str_repr = str(self.generator)
        self.assertIn("trained", str_repr.lower())


@unittest.skipIf(not TESTS_CAN_RUN, "PyTorch not available")
class TestBuildRNNModelFromFile(unittest.TestCase):
    """Test the build_rnn_model_from_file function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_file = tempfile.NamedTemporaryFile(
            mode='w',
            delete=False,
            suffix='.txt',
            encoding='utf-8'
        )

        # Write sample content
        self.temp_file.write('=' * 80 + '\n')
        self.temp_file.write('POST 1:\n')
        self.temp_file.write('Reading Time: 5 minutes\n')
        self.temp_file.write(('Sample content for RNN training. ' * 50) + '\n')
        self.temp_file.close()

    def tearDown(self):
        """Clean up test fixtures."""
        os.unlink(self.temp_file.name)

    def test_build_model(self):
        """Test building a model from file."""
        model = build_rnn_model_from_file(
            self.temp_file.name,
            embedding_dim=32,
            hidden_dim=64,
            num_layers=1,
            num_epochs=1
        )

        self.assertIsInstance(model, RNNModel)
        self.assertEqual(model.embedding_dim, 32)
        self.assertIsNotNone(model.model)


if __name__ == '__main__':
    unittest.main()
