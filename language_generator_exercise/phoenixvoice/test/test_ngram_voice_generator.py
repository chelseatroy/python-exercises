"""
Unit tests for N-gram Voice Generator.

Tests the n-gram model with backoff and smoothing functionality.
"""

import unittest
import tempfile
import os
from phoenixvoice.src.ngram_voice_generator import (
    NgramModel,
    NgramVoiceGenerator,
    build_ngram_model_from_file
)


class TestNgramModel(unittest.TestCase):
    """Test the NgramModel class."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = NgramModel(max_order=2, smoothing_alpha=0.1)

    def test_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.model.max_order, 2)
        self.assertEqual(self.model.smoothing_alpha, 0.1)
        self.assertEqual(len(self.model.vocabulary), 0)
        self.assertIn(1, self.model.models)
        self.assertIn(2, self.model.models)

    def test_add_post(self):
        """Test adding a post to the model."""
        text = "The quick brown fox jumps over the lazy dog"
        self.model.add_post(text)

        # Check vocabulary was updated
        self.assertGreater(len(self.model.vocabulary), 0)

        # Check transitions were created
        self.assertGreater(len(self.model.models[1]['transitions']), 0)
        self.assertGreater(len(self.model.models[2]['transitions']), 0)

    def test_tokenization(self):
        """Test tokenization."""
        text = "Hello, world! How are you?"
        tokens = self.model._tokenize(text)

        self.assertIn("Hello", tokens)
        self.assertIn(",", tokens)
        self.assertIn("world", tokens)
        self.assertIn("!", tokens)

    def test_backoff_probability(self):
        """Test probability calculation with backoff."""
        # Add training data
        self.model.add_post("the cat sat on the mat")
        self.model.add_post("the dog sat on the rug")

        # Test existing bigram
        prob = self.model.get_probability_with_backoff(("the",), "cat")
        self.assertGreater(prob, 0)

        # Test unseen bigram (should backoff)
        prob = self.model.get_probability_with_backoff(("xyz",), "abc")
        self.assertGreater(prob, 0)  # Should still return non-zero due to smoothing

    def test_generate_text(self):
        """Test text generation."""
        # Add enough data
        for _ in range(5):
            self.model.add_post("The quick brown fox jumps over the lazy dog.")

        text = self.model.generate_text(max_tokens=20)

        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)

    def test_detokenization(self):
        """Test converting tokens back to text."""
        tokens = ["Hello", ",", "world", "!"]
        text = self.model._detokenize(tokens)

        self.assertEqual(text, "Hello, world!")

    def test_model_length(self):
        """Test __len__ returns total n-grams."""
        self.model.add_post("The cat sat on the mat")

        length = len(self.model)
        self.assertGreater(length, 0)

    def test_save_and_load(self):
        """Test saving and loading the model."""
        # Train model
        self.model.add_post("The quick brown fox jumps over the lazy dog")

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.pkl') as f:
            temp_path = f.name

        try:
            self.model.save(temp_path)

            # Load model
            loaded_model = NgramModel.load(temp_path)

            # Verify
            self.assertEqual(loaded_model.max_order, self.model.max_order)
            self.assertEqual(len(loaded_model.vocabulary), len(self.model.vocabulary))

        finally:
            os.unlink(temp_path)


class TestNgramVoiceGenerator(unittest.TestCase):
    """Test the NgramVoiceGenerator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.generator = NgramVoiceGenerator(max_order=2, smoothing_alpha=0.1)

        # Create a temporary training file
        self.temp_file = tempfile.NamedTemporaryFile(
            mode='w',
            delete=False,
            suffix='.txt',
            encoding='utf-8'
        )

        # Write sample blog posts
        self.temp_file.write('=' * 80 + '\n')
        self.temp_file.write('POST 1: Sample Post\n')
        self.temp_file.write('Date: 2024-01-01\n')
        self.temp_file.write('URL: http://example.com/post1\n')
        self.temp_file.write('Reading Time: 5 minutes\n\n')
        self.temp_file.write('This is a sample blog post about machine learning. ' * 10)
        self.temp_file.write('\n\n')
        self.temp_file.write('=' * 80 + '\n')
        self.temp_file.write('POST 2: Another Post\n')
        self.temp_file.write('Date: 2024-01-02\n')
        self.temp_file.write('URL: http://example.com/post2\n')
        self.temp_file.write('Reading Time: 3 minutes\n\n')
        self.temp_file.write('This post discusses natural language processing. ' * 10)
        self.temp_file.close()

    def tearDown(self):
        """Clean up test fixtures."""
        os.unlink(self.temp_file.name)

    def test_initialization(self):
        """Test generator initialization."""
        self.assertEqual(self.generator.max_order, 2)
        self.assertEqual(self.generator.smoothing_alpha, 0.1)
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

        text = self.generator.generate_text(max_tokens=30, num_samples=1)

        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)

    def test_generate_multiple_samples(self):
        """Test generating multiple text samples."""
        self.generator.train(self.temp_file.name)

        texts = self.generator.generate_text(max_tokens=30, num_samples=3)

        self.assertIsInstance(texts, list)
        self.assertEqual(len(texts), 3)
        for text in texts:
            self.assertIsInstance(text, str)
            self.assertGreater(len(text), 0)

    def test_save_and_load(self):
        """Test saving and loading the generator."""
        # Train generator
        self.generator.train(self.temp_file.name)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.pkl') as f:
            save_path = f.name

        try:
            self.generator.save(save_path)

            # Create new generator and load
            new_generator = NgramVoiceGenerator()
            new_generator.load(save_path)

            # Verify
            self.assertTrue(new_generator.is_trained)
            self.assertEqual(new_generator.max_order, self.generator.max_order)

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

    def test_callable_interface(self):
        """Test that generator can be called like a function."""
        self.generator.train(self.temp_file.name)

        # Test __call__ method
        text = self.generator(max_tokens=30)

        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)


class TestBuildNgramModelFromFile(unittest.TestCase):
    """Test the build_ngram_model_from_file function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_file = tempfile.NamedTemporaryFile(
            mode='w',
            delete=False,
            suffix='.txt',
            encoding='utf-8'
        )

        # Write sample blog posts
        self.temp_file.write('=' * 80 + '\n')
        self.temp_file.write('POST 1:\n')
        self.temp_file.write('Reading Time: 5 minutes\n')
        self.temp_file.write('Sample content here. ' * 20)
        self.temp_file.close()

    def tearDown(self):
        """Clean up test fixtures."""
        os.unlink(self.temp_file.name)

    def test_build_model(self):
        """Test building a model from file."""
        model = build_ngram_model_from_file(self.temp_file.name, max_order=2)

        self.assertIsInstance(model, NgramModel)
        self.assertEqual(model.max_order, 2)
        self.assertGreater(len(model.vocabulary), 0)


if __name__ == '__main__':
    unittest.main()
