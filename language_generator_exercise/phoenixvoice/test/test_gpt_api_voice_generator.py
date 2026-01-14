"""
Unit tests for GPT API Voice Generator.

Tests the OpenAI API-based text generation functionality.
Note: These tests require the openai package and will be mocked to avoid actual API calls.
"""

import unittest
import tempfile
import os
import json
from unittest.mock import Mock, patch, MagicMock

# Check if openai is available
try:
    from phoenixvoice.src.gpt_api_voice_generator import (
        GPTAPIModel,
        GPTAPIVoiceGenerator,
        build_gpt_model_from_file,
        OPENAI_AVAILABLE
    )
    TESTS_CAN_RUN = OPENAI_AVAILABLE
except ImportError:
    TESTS_CAN_RUN = False
    GPTAPIModel = None
    GPTAPIVoiceGenerator = None


@unittest.skipIf(not TESTS_CAN_RUN, "openai package not available")
class TestGPTAPIModel(unittest.TestCase):
    """Test the GPTAPIModel class with mocked API calls."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock the OpenAI API key
        self.api_key_patch = patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key-12345'})
        self.api_key_patch.start()

    def tearDown(self):
        """Clean up patches."""
        self.api_key_patch.stop()

    @patch('phoenixvoice.src.gpt_api_voice_generator.OpenAI')
    def test_initialization(self, mock_openai):
        """Test model initialization."""
        model = GPTAPIModel(model="gpt-4o-mini-2024-07-18", temperature=0.8)

        self.assertEqual(model.base_model, "gpt-4o-mini-2024-07-18")
        self.assertEqual(model.temperature, 0.8)
        self.assertIsNone(model.fine_tuned_model)

    def test_initialization_without_api_key(self):
        """Test that initialization fails without API key."""
        # Remove API key
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                GPTAPIModel()

    @patch('phoenixvoice.src.gpt_api_voice_generator.OpenAI')
    def test_prepare_training_data(self, mock_openai):
        """Test preparing training data in JSONL format."""
        model = GPTAPIModel()

        posts = [
            "This is a sample blog post about machine learning. " * 10,
            "Another post discussing natural language processing. " * 10
        ]

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
            temp_path = f.name

        try:
            output_file = model.prepare_training_data(posts, output_file=temp_path)

            # Verify file was created
            self.assertTrue(os.path.exists(output_file))

            # Verify JSONL format
            with open(output_file, 'r') as f:
                lines = f.readlines()
                self.assertGreater(len(lines), 0)

                # Check first line is valid JSON
                first_example = json.loads(lines[0])
                self.assertIn('messages', first_example)
                self.assertIsInstance(first_example['messages'], list)

        finally:
            os.unlink(temp_path)

    @patch('phoenixvoice.src.gpt_api_voice_generator.OpenAI')
    def test_upload_training_data(self, mock_openai):
        """Test uploading training data to OpenAI."""
        # Mock the API response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.id = 'file-12345'
        mock_client.files.create.return_value = mock_response
        mock_openai.return_value = mock_client

        model = GPTAPIModel()

        # Create temporary training file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
            f.write('{"messages": [{"role": "user", "content": "test"}]}\n')
            temp_path = f.name

        try:
            file_id = model.upload_training_data(temp_path)

            self.assertEqual(file_id, 'file-12345')
            self.assertEqual(model.training_file_id, 'file-12345')
            mock_client.files.create.assert_called_once()

        finally:
            os.unlink(temp_path)

    @patch('phoenixvoice.src.gpt_api_voice_generator.OpenAI')
    def test_start_fine_tuning(self, mock_openai):
        """Test starting a fine-tuning job."""
        # Mock the API response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.id = 'ftjob-12345'
        mock_client.fine_tuning.jobs.create.return_value = mock_response
        mock_openai.return_value = mock_client

        model = GPTAPIModel()
        model.training_file_id = 'file-12345'

        job_id = model.start_fine_tuning()

        self.assertEqual(job_id, 'ftjob-12345')
        mock_client.fine_tuning.jobs.create.assert_called_once()

    @patch('phoenixvoice.src.gpt_api_voice_generator.OpenAI')
    def test_generate_text(self, mock_openai):
        """Test text generation with fine-tuned model."""
        # Mock the API response
        mock_client = Mock()
        mock_message = Mock()
        mock_message.content = "Generated text here"
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        model = GPTAPIModel()
        model.fine_tuned_model = 'ft:gpt-4o-mini:org:model:12345'

        text = model.generate_text(prompt="Write something")

        self.assertEqual(text, "Generated text here")
        mock_client.chat.completions.create.assert_called_once()

    @patch('phoenixvoice.src.gpt_api_voice_generator.OpenAI')
    def test_save_and_load(self, mock_openai):
        """Test saving and loading model configuration."""
        model = GPTAPIModel(model="gpt-4o-mini-2024-07-18", temperature=0.9)
        model.fine_tuned_model = 'ft:gpt-4o-mini:org:model:12345'
        model.training_file_id = 'file-12345'

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name

        try:
            model.save(temp_path)

            # Load model
            loaded_model = GPTAPIModel.load(temp_path)

            # Verify
            self.assertEqual(loaded_model.fine_tuned_model, model.fine_tuned_model)
            self.assertEqual(loaded_model.base_model, model.base_model)
            self.assertEqual(loaded_model.temperature, model.temperature)

        finally:
            os.unlink(temp_path)


@unittest.skipIf(not TESTS_CAN_RUN, "openai package not available")
class TestGPTAPIVoiceGenerator(unittest.TestCase):
    """Test the GPTAPIVoiceGenerator class."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock the OpenAI API key
        self.api_key_patch = patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key-12345'})
        self.api_key_patch.start()

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
        self.temp_file.write('Reading Time: 5 minutes\n\n')
        self.temp_file.write('Sample content here. ' * 20)
        self.temp_file.close()

    def tearDown(self):
        """Clean up test fixtures."""
        os.unlink(self.temp_file.name)
        self.api_key_patch.stop()

    @patch('phoenixvoice.src.gpt_api_voice_generator.OpenAI')
    def test_initialization(self, mock_openai):
        """Test generator initialization."""
        generator = GPTAPIVoiceGenerator(
            model="gpt-4o-mini-2024-07-18",
            temperature=0.8
        )

        self.assertEqual(generator.model_name, "gpt-4o-mini-2024-07-18")
        self.assertEqual(generator.temperature, 0.8)
        self.assertFalse(generator.is_trained)

    @patch('phoenixvoice.src.gpt_api_voice_generator.build_gpt_model_from_file')
    @patch('phoenixvoice.src.gpt_api_voice_generator.OpenAI')
    def test_train(self, mock_openai, mock_build):
        """Test training the generator (mocked)."""
        # Mock the build function to return a trained model
        mock_model = Mock()
        mock_model.fine_tuned_model = 'ft:gpt-4o-mini:org:model:12345'
        mock_build.return_value = mock_model

        generator = GPTAPIVoiceGenerator()
        generator.train(self.temp_file.name)

        self.assertTrue(generator.is_trained)
        self.assertIsNotNone(generator.model)
        mock_build.assert_called_once()

    @patch('phoenixvoice.src.gpt_api_voice_generator.OpenAI')
    def test_generate_text_before_training(self, mock_openai):
        """Test that generating before training raises an error."""
        generator = GPTAPIVoiceGenerator()

        with self.assertRaises(RuntimeError):
            generator.generate_text()

    @patch('phoenixvoice.src.gpt_api_voice_generator.OpenAI')
    def test_generate_single_sample(self, mock_openai):
        """Test generating a single text sample."""
        # Mock the model
        mock_model = Mock()
        mock_model.generate_text.return_value = "Generated sample text"

        generator = GPTAPIVoiceGenerator()
        generator.model = mock_model
        generator._is_trained = True

        text = generator.generate_text(max_tokens=100, num_samples=1)

        self.assertEqual(text, "Generated sample text")
        self.assertIsInstance(text, str)

    @patch('phoenixvoice.src.gpt_api_voice_generator.OpenAI')
    def test_generate_multiple_samples(self, mock_openai):
        """Test generating multiple text samples."""
        # Mock the model
        mock_model = Mock()
        mock_model.generate_text.side_effect = ["Sample 1", "Sample 2", "Sample 3"]

        generator = GPTAPIVoiceGenerator()
        generator.model = mock_model
        generator._is_trained = True

        texts = generator.generate_text(max_tokens=100, num_samples=3)

        self.assertIsInstance(texts, list)
        self.assertEqual(len(texts), 3)
        self.assertEqual(texts, ["Sample 1", "Sample 2", "Sample 3"])

    @patch('phoenixvoice.src.gpt_api_voice_generator.OpenAI')
    def test_generate_with_prompt(self, mock_openai):
        """Test generation with custom prompt."""
        mock_model = Mock()
        mock_model.generate_text.return_value = "Generated with prompt"

        generator = GPTAPIVoiceGenerator()
        generator.model = mock_model
        generator._is_trained = True

        text = generator.generate_text(prompt="Write about ML", num_samples=1)

        mock_model.generate_text.assert_called_with(prompt="Write about ML", max_tokens=None)
        self.assertEqual(text, "Generated with prompt")

    @patch('phoenixvoice.src.gpt_api_voice_generator.OpenAI')
    def test_save_and_load(self, mock_openai):
        """Test saving and loading the generator."""
        # Create a mock model
        mock_model = Mock()
        mock_model.fine_tuned_model = 'ft:gpt-4o-mini:org:model:12345'
        mock_model.base_model = 'gpt-4o-mini-2024-07-18'
        mock_model.temperature = 0.8
        mock_model.max_tokens = 500
        mock_model.training_file_id = 'file-12345'

        generator = GPTAPIVoiceGenerator()
        generator.model = mock_model
        generator._is_trained = True

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            save_path = f.name

        try:
            # Mock the save method
            mock_model.save = Mock()

            generator.save(save_path)

            mock_model.save.assert_called_once_with(save_path)

        finally:
            try:
                os.unlink(save_path)
            except FileNotFoundError:
                pass

    @patch('phoenixvoice.src.gpt_api_voice_generator.OpenAI')
    def test_string_representation(self, mock_openai):
        """Test __str__ and __repr__ methods."""
        generator = GPTAPIVoiceGenerator()

        str_repr = str(generator)
        self.assertIn("untrained", str_repr.lower())

        # Mock training
        generator._is_trained = True

        str_repr = str(generator)
        self.assertIn("trained", str_repr.lower())


@unittest.skipIf(not TESTS_CAN_RUN, "openai package not available")
class TestBuildGPTModelFromFile(unittest.TestCase):
    """Test the build_gpt_model_from_file function."""

    def setUp(self):
        """Set up test fixtures."""
        self.api_key_patch = patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key-12345'})
        self.api_key_patch.start()

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
        self.temp_file.write('Sample content here. ' * 20)
        self.temp_file.close()

    def tearDown(self):
        """Clean up test fixtures."""
        os.unlink(self.temp_file.name)
        self.api_key_patch.stop()

    @patch('phoenixvoice.src.gpt_api_voice_generator.OpenAI')
    @patch('phoenixvoice.src.gpt_api_voice_generator.GPTAPIModel.train_from_posts')
    def test_build_model(self, mock_train, mock_openai):
        """Test building a model from file (mocked to avoid API calls)."""
        mock_train.return_value = None

        model = build_gpt_model_from_file(
            self.temp_file.name,
            model="gpt-4o-mini-2024-07-18",
            temperature=0.8
        )

        self.assertIsInstance(model, GPTAPIModel)
        self.assertEqual(model.base_model, "gpt-4o-mini-2024-07-18")
        mock_train.assert_called_once()


if __name__ == '__main__':
    unittest.main()
