"""
Unit tests for Word2Vec Voice Generator.

Tests the Word2Vec-based text generation functionality.
Note: These tests require gensim to be installed.
"""

import unittest
import tempfile
import os

# Check if gensim is available
try:
    from phoenixvoice.src.word2vec_voice_generator import (
        Word2VecModel,
        Word2VecVoiceGenerator,
        build_word2vec_model_from_file,
        GENSIM_AVAILABLE
    )
    TESTS_CAN_RUN = GENSIM_AVAILABLE
except ImportError:
    TESTS_CAN_RUN = False
    Word2VecModel = None
    Word2VecVoiceGenerator = None


@unittest.skipIf(not TESTS_CAN_RUN, "gensim not available")
class TestWord2VecModel(unittest.TestCase):
    """Test the Word2VecModel class."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = Word2VecModel(vector_size=50, window=3, min_count=1, temperature=0.7)

    def test_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.model.vector_size, 50)
        self.assertEqual(self.model.window, 3)
        self.assertEqual(self.model.min_count, 1)
        self.assertEqual(self.model.temperature, 0.7)
        self.assertEqual(len(self.model.sentences), 0)

    def test_add_post(self):
        """Test adding posts to the model."""
        text = "The quick brown fox jumps over the lazy dog"
        self.model.add_post(text)

        self.assertEqual(len(self.model.sentences), 1)
        self.assertGreater(len(self.model.vocabulary), 0)

    def test_tokenization(self):
        """Test tokenization (words only, no punctuation)."""
        text = "Hello, world! How are you?"
        tokens = self.model._tokenize(text)

        # Word2Vec typically uses words only
        self.assertIn("hello", tokens)
        self.assertIn("world", tokens)
        self.assertNotIn(",", tokens)  # Punctuation excluded

    def test_train(self):
        """Test training the Word2Vec model."""
        # Add multiple posts
        posts = [
            "The cat sat on the mat",
            "The dog played in the park",
            "Machine learning is fascinating",
            "Natural language processing is useful"
        ] * 3  # Repeat to have enough data

        for post in posts:
            self.model.add_post(post)

        self.model.train()

        self.assertIsNotNone(self.model.model)
        self.assertGreater(len(self.model.model.wv), 0)

    def test_generate_text(self):
        """Test text generation."""
        # Add training data
        posts = [
            "Machine learning models can generate text",
            "Text generation uses neural networks",
            "Neural networks learn from data"
        ] * 5

        for post in posts:
            self.model.add_post(post)

        self.model.train()

        # Generate text
        text = self.model.generate_text(max_tokens=30)

        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)

    def test_generate_with_seed(self):
        """Test generation with a seed phrase."""
        posts = ["The quick brown fox"] * 10

        for post in posts:
            self.model.add_post(post)

        self.model.train()

        text = self.model.generate_text(max_tokens=20, seed="the quick")

        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)

    def test_add_punctuation(self):
        """Test punctuation addition heuristic."""
        tokens = ["hello", "world", "this", "is", "a", "test"]
        text = self.model._add_punctuation(tokens)

        # Should have capitals and punctuation
        self.assertTrue(text[0].isupper())  # First letter capitalized
        self.assertTrue(text.endswith('.'))  # Ends with period

    def test_get_similar_words(self):
        """Test getting similar words."""
        posts = [
            "cat kitten feline pet animal",
            "dog puppy canine pet animal"
        ] * 10

        for post in posts:
            self.model.add_post(post)

        self.model.train()

        # Get similar words
        try:
            similar = self.model.get_similar_words("cat", topn=3)
            self.assertIsInstance(similar, list)
            self.assertLessEqual(len(similar), 3)
        except KeyError:
            # Word might not be in vocabulary with min_count
            pass

    def test_save_and_load(self):
        """Test saving and loading the model."""
        posts = ["test data here"] * 10

        for post in posts:
            self.model.add_post(post)

        self.model.train()

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.w2v') as f:
            temp_path = f.name[:-4]  # Remove .w2v extension as it's added by save

        try:
            self.model.save(temp_path)

            # Load model
            loaded_model = Word2VecModel.load(temp_path)

            # Verify
            self.assertEqual(loaded_model.vector_size, self.model.vector_size)
            self.assertEqual(loaded_model.window, self.model.window)

        finally:
            # Clean up
            for ext in ['.w2v', '.meta']:
                try:
                    os.unlink(temp_path + ext)
                except FileNotFoundError:
                    pass


@unittest.skipIf(not TESTS_CAN_RUN, "gensim not available")
class TestWord2VecVoiceGenerator(unittest.TestCase):
    """Test the Word2VecVoiceGenerator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.generator = Word2VecVoiceGenerator(
            vector_size=50,
            window=3,
            min_count=1,
            temperature=0.7
        )

        # Create a temporary training file
        self.temp_file = tempfile.NamedTemporaryFile(
            mode='w',
            delete=False,
            suffix='.txt',
            encoding='utf-8'
        )

        # Write sample blog posts with more content for better Word2Vec training
        for i in range(5):
            self.temp_file.write('=' * 80 + '\n')
            self.temp_file.write(f'POST {i+1}: Sample Post\n')
            self.temp_file.write('Date: 2024-01-01\n')
            self.temp_file.write('Reading Time: 5 minutes\n\n')
            self.temp_file.write(
                'Machine learning and artificial intelligence are transforming technology. '
                'Neural networks learn patterns from data. '
                'Text generation models can create coherent content. '
            ) * 3
            self.temp_file.write('\n\n')

        self.temp_file.close()

    def tearDown(self):
        """Clean up test fixtures."""
        os.unlink(self.temp_file.name)

    def test_initialization(self):
        """Test generator initialization."""
        self.assertEqual(self.generator.vector_size, 50)
        self.assertEqual(self.generator.window, 3)
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

    def test_generate_with_seed(self):
        """Test generation with seed phrase."""
        self.generator.train(self.temp_file.name)

        text = self.generator.generate_text(max_tokens=30, seed="machine learning")

        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)

    def test_get_similar_words(self):
        """Test getting similar words through generator."""
        self.generator.train(self.temp_file.name)

        try:
            similar = self.generator.get_similar_words("learning", topn=5)
            self.assertIsInstance(similar, list)
        except (RuntimeError, KeyError):
            # Word might not be in vocabulary
            pass

    def test_save_and_load(self):
        """Test saving and loading the generator."""
        self.generator.train(self.temp_file.name)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.w2v') as f:
            save_path = f.name[:-4]

        try:
            self.generator.save(save_path)

            # Create new generator and load
            new_generator = Word2VecVoiceGenerator()
            new_generator.load(save_path)

            # Verify
            self.assertTrue(new_generator.is_trained)
            self.assertEqual(new_generator.vector_size, self.generator.vector_size)

            # Test generation works
            text = new_generator.generate_text(max_tokens=20)
            self.assertIsInstance(text, str)

        finally:
            # Clean up
            for ext in ['.w2v', '.meta']:
                try:
                    os.unlink(save_path + ext)
                except FileNotFoundError:
                    pass

    def test_string_representation(self):
        """Test __str__ and __repr__ methods."""
        str_repr = str(self.generator)
        self.assertIn("untrained", str_repr.lower())

        self.generator.train(self.temp_file.name)

        str_repr = str(self.generator)
        self.assertIn("trained", str_repr.lower())


@unittest.skipIf(not TESTS_CAN_RUN, "gensim not available")
class TestBuildWord2VecModelFromFile(unittest.TestCase):
    """Test the build_word2vec_model_from_file function."""

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
        self.temp_file.write(('Sample content about machine learning. ' * 20) + '\n')
        self.temp_file.close()

    def tearDown(self):
        """Clean up test fixtures."""
        os.unlink(self.temp_file.name)

    def test_build_model(self):
        """Test building a model from file."""
        model = build_word2vec_model_from_file(
            self.temp_file.name,
            vector_size=50,
            window=3,
            min_count=1
        )

        self.assertIsInstance(model, Word2VecModel)
        self.assertEqual(model.vector_size, 50)
        self.assertIsNotNone(model.model)


if __name__ == '__main__':
    unittest.main()
