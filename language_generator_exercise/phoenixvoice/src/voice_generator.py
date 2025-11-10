from abc import ABC, abstractmethod


class VoiceGenerator(ABC):
    """
    Abstract base class for text generation models that learn to mimic a writing voice.

    This interface allows switching between different generation methods (Markov, transformer, etc.)
    while maintaining consistent API.
    """

    def __init__(self):
        self._is_trained = False

    @property
    def is_trained(self):
        """Returns True if the model has been trained or loaded."""
        return self._is_trained

    @abstractmethod
    def train(self, text_file_path):
        """
        Train the model on text from a file.

        Args:
            text_file_path: Path to text file containing training data
        """
        pass

    @abstractmethod
    def generate_text(self, max_tokens=50, num_samples=1):
        """
        Generate text samples using the trained model.

        Args:
            max_tokens: Maximum number of tokens to generate
            num_samples: Number of text samples to generate

        Returns:
            If num_samples=1: string of generated text
            If num_samples>1: list of generated text strings
        """
        pass

    @abstractmethod
    def save(self, filepath):
        """
        Save the trained model to disk.

        Args:
            filepath: Path where model should be saved
        """
        pass

    @abstractmethod
    def load(self, filepath):
        """
        Load a previously trained model from disk.

        Args:
            filepath: Path to saved model file
        """
        pass

    def __str__(self):
        """
        Return a human-readable string representation of the generator.

        Returns:
            String showing the class name and training status
        """
        status = "trained" if self.is_trained else "untrained"
        return f"{self.__class__.__name__}({status})"

    def __repr__(self):
        """
        Return a developer-friendly representation of the generator.

        Returns:
            String showing the class name and is_trained property
        """
        return f"{self.__class__.__name__}(is_trained={self.is_trained})"

    def __call__(self, max_tokens=50, num_samples=1):
        """
        Allow calling the generator directly to generate text.

        This makes the generator behave like a function, which is a common
        pattern for ML models.

        Args:
            max_tokens: Maximum number of tokens to generate
            num_samples: Number of text samples to generate

        Returns:
            Generated text (string if num_samples=1, list otherwise)

        Example:
            generator = create_voice_generator("markov", order=2)
            generator.train("blog_posts.txt")
            text = generator(max_tokens=100, num_samples=3)
        """
        return self.generate_text(max_tokens=max_tokens, num_samples=num_samples)
