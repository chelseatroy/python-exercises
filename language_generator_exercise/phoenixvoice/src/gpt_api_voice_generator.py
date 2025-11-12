"""
GPT API-based voice generator.

This uses OpenAI's API to fine-tune a GPT model on blog posts,
providing production-quality text generation without local compute requirements.

Good for learning about:
- Fine-tuning vs. training from scratch
- API-based ML services
- Cost-performance trade-offs
- Comparing local vs. cloud approaches

Installation required:
    pip install openai

Setup required:
    export OPENAI_API_KEY="your-api-key"
    Or create a .env file with: OPENAI_API_KEY=your-api-key
"""

import os
import re
import json
import time
from pathlib import Path
from .voice_generator import VoiceGenerator

try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None


class GPTAPIModel:
    """
    GPT model using OpenAI's fine-tuning API.

    This creates a fine-tuned model that learns your writing style
    and can generate text in that style on demand.
    """

    def __init__(self, model="gpt-4o-mini-2024-07-18", temperature=0.8, max_tokens=500):
        """
        Initialize GPT API model.

        Args:
            model: Base model to fine-tune (default: gpt-4o-mini-2024-07-18)
            temperature: Sampling temperature (default: 0.8)
            max_tokens: Maximum tokens to generate (default: 500)
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "openai package is required for GPT API generation. "
                "Install with: pip install openai"
            )

        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable not set. "
                "Set it with: export OPENAI_API_KEY='your-key'"
            )

        self.client = OpenAI(api_key=api_key)
        self.base_model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.fine_tuned_model = None
        self.training_file_id = None

    def prepare_training_data(self, posts, output_file="training_data.jsonl"):
        """
        Prepare blog posts for fine-tuning format.

        OpenAI expects JSONL format with "messages" containing conversation turns.
        For style mimicry, we create examples where the system is asked to write
        in the style, and the assistant responds with actual blog content.

        Args:
            posts: List of blog post texts
            output_file: Path to save training data

        Returns:
            Path to training data file
        """
        training_examples = []

        for post in posts:
            # Split long posts into paragraphs for more training examples
            paragraphs = [p.strip() for p in post.split('\n\n') if p.strip()]

            for paragraph in paragraphs:
                # Skip very short paragraphs
                if len(paragraph) < 100:
                    continue

                # Create a training example
                example = {
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a blog writer. Write in a clear, engaging style."
                        },
                        {
                            "role": "user",
                            "content": "Write a paragraph about technical topics in an educational style."
                        },
                        {
                            "role": "assistant",
                            "content": paragraph
                        }
                    ]
                }
                training_examples.append(example)

        # Write to JSONL file
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in training_examples:
                f.write(json.dumps(example) + '\n')

        print(f"Prepared {len(training_examples)} training examples")
        print(f"Training data saved to {output_file}")

        return output_file

    def upload_training_data(self, file_path):
        """
        Upload training data to OpenAI.

        Args:
            file_path: Path to training data file

        Returns:
            File ID from OpenAI
        """
        print(f"Uploading training data from {file_path}...")

        with open(file_path, 'rb') as f:
            response = self.client.files.create(
                file=f,
                purpose='fine-tune'
            )

        self.training_file_id = response.id
        print(f"Training data uploaded with ID: {self.training_file_id}")

        return self.training_file_id

    def start_fine_tuning(self):
        """
        Start fine-tuning job on OpenAI.

        Returns:
            Fine-tuning job ID
        """
        if not self.training_file_id:
            raise ValueError("Must upload training data first")

        print(f"Starting fine-tuning job with base model: {self.base_model}")

        response = self.client.fine_tuning.jobs.create(
            training_file=self.training_file_id,
            model=self.base_model
        )

        job_id = response.id
        print(f"Fine-tuning job started with ID: {job_id}")
        print("This may take 10-60 minutes depending on data size...")

        return job_id

    def wait_for_fine_tuning(self, job_id):
        """
        Wait for fine-tuning job to complete.

        Args:
            job_id: Fine-tuning job ID

        Returns:
            Fine-tuned model ID
        """
        print("Waiting for fine-tuning to complete...")
        print("You can monitor progress at: https://platform.openai.com/finetune")

        while True:
            response = self.client.fine_tuning.jobs.retrieve(job_id)
            status = response.status

            print(f"Status: {status}")

            if status == 'succeeded':
                self.fine_tuned_model = response.fine_tuned_model
                print(f"Fine-tuning completed! Model: {self.fine_tuned_model}")
                return self.fine_tuned_model

            elif status == 'failed':
                raise RuntimeError(f"Fine-tuning failed: {response.error}")

            elif status in ['cancelled', 'expired']:
                raise RuntimeError(f"Fine-tuning {status}")

            # Wait before checking again
            time.sleep(30)

    def train_from_posts(self, posts):
        """
        Complete training pipeline: prepare data, upload, fine-tune.

        Args:
            posts: List of blog post texts
        """
        # Prepare training data
        training_file = self.prepare_training_data(posts)

        # Upload to OpenAI
        self.upload_training_data(training_file)

        # Start fine-tuning
        job_id = self.start_fine_tuning()

        # Wait for completion
        self.wait_for_fine_tuning(job_id)

        print("Training complete!")

    def generate_text(self, prompt=None, max_tokens=None):
        """
        Generate text using the fine-tuned model.

        Args:
            prompt: Optional prompt to guide generation
            max_tokens: Optional override for max tokens

        Returns:
            Generated text as a string
        """
        if not self.fine_tuned_model:
            raise RuntimeError("Model must be fine-tuned before generating text")

        if prompt is None:
            prompt = "Write a paragraph about technical topics in an educational style."

        if max_tokens is None:
            max_tokens = self.max_tokens

        response = self.client.chat.completions.create(
            model=self.fine_tuned_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a blog writer. Write in a clear, engaging style."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=self.temperature,
            max_tokens=max_tokens
        )

        return response.choices[0].message.content

    def save(self, filepath):
        """
        Save model information to disk.

        Note: The actual model is stored on OpenAI's servers.
        This just saves the model ID and configuration.
        """
        if not self.fine_tuned_model:
            raise RuntimeError("Model must be fine-tuned before saving")

        config = {
            'fine_tuned_model': self.fine_tuned_model,
            'base_model': self.base_model,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'training_file_id': self.training_file_id
        }

        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"Model configuration saved to {filepath}")
        print(f"Fine-tuned model ID: {self.fine_tuned_model}")

    @staticmethod
    def load(filepath):
        """
        Load model information from disk.

        Args:
            filepath: Path to saved configuration

        Returns:
            GPTAPIModel instance with loaded configuration
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "openai package is required. "
                "Install with: pip install openai"
            )

        with open(filepath, 'r') as f:
            config = json.load(f)

        model = GPTAPIModel(
            model=config['base_model'],
            temperature=config['temperature'],
            max_tokens=config['max_tokens']
        )

        model.fine_tuned_model = config['fine_tuned_model']
        model.training_file_id = config.get('training_file_id')

        print(f"Model configuration loaded from {filepath}")
        print(f"Using fine-tuned model: {model.fine_tuned_model}")

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


def build_gpt_model_from_file(filename, **kwargs):
    """
    Build a GPT API model from a blog posts file.

    Args:
        filename: Path to the blog posts text file
        **kwargs: Arguments to pass to GPTAPIModel constructor

    Returns:
        Trained GPTAPIModel instance
    """
    posts = parse_blog_posts_file(filename)

    print(f"Building GPT model from {len(posts)} posts")

    model = GPTAPIModel(**kwargs)
    model.train_from_posts(posts)

    return model


class GPTAPIVoiceGenerator(VoiceGenerator):
    """
    VoiceGenerator implementation using OpenAI's GPT fine-tuning API.

    This is the most production-ready approach:
    - Uses state-of-the-art transformer models
    - No local compute required (training happens on OpenAI's servers)
    - High-quality, coherent text generation
    - Cost-effective for inference (pay per use)

    Pedagogical value:
    - Understanding API-based ML services
    - Comparing cloud vs. local approaches
    - Experiencing production ML workflows
    - Learning about fine-tuning vs. training from scratch
    - Understanding cost-performance trade-offs

    Cost considerations:
    - Fine-tuning: ~$0.008 per 1K tokens (one-time)
    - Inference: ~$0.0015 per 1K tokens (per generation)
    - For 734K words (~1M tokens): ~$8 for training, $1.50 per 1M tokens generated
    """

    def __init__(self, model="gpt-4o-mini-2024-07-18", temperature=0.8, max_tokens=500):
        """
        Initialize a GPT API voice generator.

        Args:
            model: Base model to fine-tune (default: gpt-4o-mini-2024-07-18)
            temperature: Sampling temperature (default: 0.8)
            max_tokens: Maximum tokens per generation (default: 500)
        """
        super().__init__()

        if not OPENAI_AVAILABLE:
            raise ImportError(
                "openai package is required for GPT API generation. "
                "Install with: pip install openai"
            )

        self.model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model = None

    def train(self, text_file_path):
        """
        Fine-tune a GPT model on text from a file.

        This will:
        1. Prepare training data in OpenAI's format
        2. Upload to OpenAI
        3. Start fine-tuning job
        4. Wait for completion (10-60 minutes)

        Note: This requires OPENAI_API_KEY environment variable to be set.
        """
        self.model = build_gpt_model_from_file(
            text_file_path,
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        self._is_trained = True

    def generate_text(self, max_tokens=None, num_samples=1, prompt=None):
        """
        Generate text samples using the fine-tuned GPT model.

        Args:
            max_tokens: Maximum tokens to generate (uses constructor value if None)
            num_samples: Number of text samples to generate
            prompt: Optional prompt to guide generation

        Returns:
            If num_samples=1: string of generated text
            If num_samples>1: list of generated text strings
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before generating text. Call train() first.")

        samples = []
        for _ in range(num_samples):
            text = self.model.generate_text(prompt=prompt, max_tokens=max_tokens)
            samples.append(text)

        return samples[0] if num_samples == 1 else samples

    def save(self, filepath):
        """
        Save the model configuration to disk.

        Note: The actual fine-tuned model lives on OpenAI's servers.
        This saves the model ID and configuration so you can use it later.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before saving. Call train() first.")

        self.model.save(filepath)

    def load(self, filepath):
        """
        Load a previously fine-tuned model configuration from disk.

        Args:
            filepath: Path to saved configuration file
        """
        self.model = GPTAPIModel.load(filepath)
        self.model_name = self.model.base_model
        self.temperature = self.model.temperature
        self.max_tokens = self.model.max_tokens
        self._is_trained = True
