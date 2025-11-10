import unittest
import os
import tempfile
import shutil
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phoenixvoice.src.markov_voice_generator import MarkovVoiceGenerator
from phoenixvoice.src.voice_generator import VoiceGenerator


class TestMarkovVoiceGenerator(unittest.TestCase):
    """Test the MarkovVoiceGenerator wrapper class."""

    def setUp(self):
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.original_dir = os.getcwd()
        os.chdir(self.test_dir)

        # Create a test blog posts file
        self.test_content = """================================================================================
POST 1: Test Post
Date: 2024-01-15
URL: https://example.com/test
================================================================================
Reading Time: 2 minutes The quick brown fox jumps over the lazy dog. The fox is fast.


================================================================================
POST 2: Another Post
Date: 2024-01-16
URL: https://example.com/another
================================================================================
Reading Time: 1 minute The lazy dog sleeps all day long.


"""
        self.test_file = "test_posts.txt"
        with open(self.test_file, 'w', encoding='utf-8') as f:
            f.write(self.test_content)

    def tearDown(self):
        # Clean up temporary directory
        os.chdir(self.original_dir)
        shutil.rmtree(self.test_dir)

    def test_inherits_from_voice_generator(self):
        """Test that MarkovVoiceGenerator inherits from VoiceGenerator."""
        generator = MarkovVoiceGenerator(order=1)
        self.assertIsInstance(generator, VoiceGenerator)

    def test_initialization_with_default_order(self):
        """Test initialization with default order."""
        generator = MarkovVoiceGenerator()
        self.assertEqual(generator.order, 1)
        self.assertFalse(generator.is_trained)

    def test_initialization_with_custom_order(self):
        """Test initialization with custom order."""
        generator = MarkovVoiceGenerator(order=3)
        self.assertEqual(generator.order, 3)
        self.assertFalse(generator.is_trained)

    def test_train_sets_is_trained_to_true(self):
        """Test that train() sets is_trained to True."""
        generator = MarkovVoiceGenerator(order=1)
        self.assertFalse(generator.is_trained)
        generator.train(self.test_file)
        self.assertTrue(generator.is_trained)

    def test_train_creates_model(self):
        """Test that train() creates a MarkovModel."""
        generator = MarkovVoiceGenerator(order=1)
        generator.train(self.test_file)
        self.assertIsNotNone(generator.model)

    def test_train_with_different_orders(self):
        """Test training with different model orders."""
        for order in [1, 2, 3]:
            with self.subTest(order=order):
                generator = MarkovVoiceGenerator(order=order)
                generator.train(self.test_file)
                self.assertEqual(generator.model.order, order)
                self.assertTrue(generator.is_trained)

    def test_generate_text_raises_error_before_training(self):
        """Test that generate_text raises error before training."""
        generator = MarkovVoiceGenerator(order=1)
        with self.assertRaises(RuntimeError) as context:
            generator.generate_text()
        self.assertIn("must be trained", str(context.exception))

    def test_generate_text_single_sample_returns_string(self):
        """Test that generate_text with num_samples=1 returns a string."""
        generator = MarkovVoiceGenerator(order=1)
        generator.train(self.test_file)
        result = generator.generate_text(max_tokens=20, num_samples=1)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_generate_text_multiple_samples_returns_list(self):
        """Test that generate_text with num_samples>1 returns a list."""
        generator = MarkovVoiceGenerator(order=1)
        generator.train(self.test_file)
        result = generator.generate_text(max_tokens=20, num_samples=3)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)
        for sample in result:
            self.assertIsInstance(sample, str)

    def test_generate_text_respects_max_tokens(self):
        """Test that generated text respects max_tokens parameter."""
        generator = MarkovVoiceGenerator(order=1)
        generator.train(self.test_file)
        # Generate with small max_tokens
        text = generator.generate_text(max_tokens=5, num_samples=1)
        # Count tokens (words and punctuation)
        tokens = text.split()
        # Should be roughly around max_tokens (may vary slightly)
        self.assertLessEqual(len(tokens), 10)  # Generous upper bound

    def test_save_raises_error_before_training(self):
        """Test that save raises error before training."""
        generator = MarkovVoiceGenerator(order=1)
        with self.assertRaises(RuntimeError) as context:
            generator.save("model.pkl")
        self.assertIn("must be trained", str(context.exception))

    def test_save_creates_file(self):
        """Test that save creates a file."""
        generator = MarkovVoiceGenerator(order=1)
        generator.train(self.test_file)
        model_file = "test_model.pkl"
        generator.save(model_file)
        self.assertTrue(os.path.exists(model_file))

    def test_load_sets_is_trained_to_true(self):
        """Test that load sets is_trained to True."""
        # First save a model
        generator = MarkovVoiceGenerator(order=2)
        generator.train(self.test_file)
        model_file = "test_model.pkl"
        generator.save(model_file)

        # Now load it in a new generator
        new_generator = MarkovVoiceGenerator()
        self.assertFalse(new_generator.is_trained)
        new_generator.load(model_file)
        self.assertTrue(new_generator.is_trained)

    def test_load_preserves_model_order(self):
        """Test that loading a model preserves its order."""
        # Save a 2nd-order model
        generator = MarkovVoiceGenerator(order=2)
        generator.train(self.test_file)
        model_file = "test_model.pkl"
        generator.save(model_file)

        # Load it
        new_generator = MarkovVoiceGenerator()
        new_generator.load(model_file)
        self.assertEqual(new_generator.order, 2)

    def test_loaded_model_can_generate_text(self):
        """Test that a loaded model can generate text."""
        # Save a model
        generator = MarkovVoiceGenerator(order=1)
        generator.train(self.test_file)
        model_file = "test_model.pkl"
        generator.save(model_file)

        # Load and generate
        new_generator = MarkovVoiceGenerator()
        new_generator.load(model_file)
        text = new_generator.generate_text(max_tokens=20)
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)

    def test_train_then_save_then_load_workflow(self):
        """Test the complete train -> save -> load workflow."""
        # Train
        generator1 = MarkovVoiceGenerator(order=2)
        generator1.train(self.test_file)

        # Generate sample from original
        sample1 = generator1.generate_text(max_tokens=30)

        # Save
        model_file = "workflow_test.pkl"
        generator1.save(model_file)

        # Load in new generator
        generator2 = MarkovVoiceGenerator()
        generator2.load(model_file)

        # Both should be able to generate
        sample2 = generator2.generate_text(max_tokens=30)

        self.assertIsInstance(sample1, str)
        self.assertIsInstance(sample2, str)
        self.assertEqual(generator1.order, generator2.order)

    def test_multiple_train_calls_retrain_model(self):
        """Test that calling train multiple times retrains the model."""
        generator = MarkovVoiceGenerator(order=1)

        # First training
        generator.train(self.test_file)
        model1 = generator.model

        # Second training
        generator.train(self.test_file)
        model2 = generator.model

        # Should create a new model
        self.assertIsNot(model1, model2)


if __name__ == '__main__':
    unittest.main()
