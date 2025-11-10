import unittest
from unittest.mock import mock_open, patch
import sys
import os
import tempfile
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phoenixvoice.src.markov_parser import MarkovModel, parse_blog_posts_file, build_markov_model_from_file


class TestMarkovModel(unittest.TestCase):

    def setUp(self):
        self.model = MarkovModel()

    def test_tokenize_simple_text(self):
        text = "Hello world"
        tokens = self.model._tokenize(text)
        self.assertEqual(tokens, ["Hello", "world"])

    def test_tokenize_with_punctuation(self):
        text = "Hello, world!"
        tokens = self.model._tokenize(text)
        self.assertEqual(tokens, ["Hello", ",", "world", "!"])

    def test_tokenize_preserves_special_tokens(self):
        text = "<START> Hello world <END>"
        tokens = self.model._tokenize(text)
        self.assertEqual(tokens, ["<START>", "Hello", "world", "<END>"])

    def test_tokenize_multiple_punctuation(self):
        text = "Well, this is nice: yes?"
        tokens = self.model._tokenize(text)
        self.assertEqual(tokens, ["Well", ",", "this", "is", "nice", ":", "yes", "?"])

    def test_add_post_builds_transitions(self):
        self.model.add_post("Hello world")

        self.assertIn(("Hello",), self.model.transitions)
        self.assertIn("world", self.model.transitions[("Hello",)])
        self.assertEqual(self.model.transitions[("Hello",)]["world"], 1)

    def test_add_post_tracks_start_tokens(self):
        self.model.add_post("Hello world")
        self.model.add_post("Goodbye world")

        self.assertEqual(len(self.model.start_states), 2)
        self.assertIn(("Hello",), self.model.start_states)
        self.assertIn(("Goodbye",), self.model.start_states)

    def test_add_post_increments_transition_counts(self):
        self.model.add_post("Hello world")
        self.model.add_post("Hello friend")

        self.assertEqual(self.model.transitions[("Hello",)]["world"], 1)
        self.assertEqual(self.model.transitions[("Hello",)]["friend"], 1)
        self.assertEqual(self.model.total_transitions[("Hello",)], 2)

    def test_add_post_handles_empty_text(self):
        self.model.add_post("")

        self.assertEqual(len(self.model.start_states), 0)
        self.assertEqual(len(self.model.transitions), 0)

    def test_get_transition_probability_simple(self):
        self.model.add_post("Hello world")

        prob = self.model.get_transition_probability(("Hello",), "world")
        self.assertEqual(prob, 1.0)

    def test_get_transition_probability_multiple_options(self):
        self.model.add_post("Hello world")
        self.model.add_post("Hello friend")

        prob_world = self.model.get_transition_probability(("Hello",), "world")
        prob_friend = self.model.get_transition_probability(("Hello",), "friend")

        self.assertEqual(prob_world, 0.5)
        self.assertEqual(prob_friend, 0.5)

    def test_get_transition_probability_nonexistent_token(self):
        self.model.add_post("Hello world")

        prob = self.model.get_transition_probability(("Goodbye",), "world")
        self.assertEqual(prob, 0.0)

    def test_get_transition_probability_nonexistent_transition(self):
        self.model.add_post("Hello world")

        prob = self.model.get_transition_probability(("Hello",), "Goodbye")
        self.assertEqual(prob, 0.0)

    def test_get_next_token_probabilities(self):
        self.model.add_post("Hello world")
        self.model.add_post("Hello friend")
        self.model.add_post("Hello world")

        probs = self.model.get_next_token_probabilities(("Hello",))

        self.assertEqual(len(probs), 2)
        self.assertAlmostEqual(probs["world"], 2/3)
        self.assertAlmostEqual(probs["friend"], 1/3)

    def test_get_next_token_probabilities_nonexistent_token(self):
        self.model.add_post("Hello world")

        probs = self.model.get_next_token_probabilities(("Goodbye",))
        self.assertEqual(probs, {})

    def test_get_next_token_probabilities_sum_to_one(self):
        self.model.add_post("The quick brown fox")

        probs = self.model.get_next_token_probabilities(("The",))
        total_prob = sum(probs.values())

        self.assertAlmostEqual(total_prob, 1.0)

    def test_detokenize_simple_words(self):
        tokens = ["Hello", "world"]
        text = self.model._detokenize(tokens)
        self.assertEqual(text, "Hello world")

    def test_detokenize_with_punctuation(self):
        tokens = ["Hello", ",", "world", "!"]
        text = self.model._detokenize(tokens)
        self.assertEqual(text, "Hello, world!")

    def test_detokenize_preserves_special_tokens(self):
        tokens = ["<START>", "Hello", "world", "<END>"]
        text = self.model._detokenize(tokens)
        self.assertEqual(text, "<START> Hello world <END>")

    def test_generate_text_returns_string(self):
        self.model.add_post("Hello world this is a test")

        text = self.model.generate_text(max_tokens=5)
        self.assertIsInstance(text, str)

    def test_generate_text_respects_max_tokens(self):
        self.model.add_post("Hello world this is a test sentence")

        text = self.model.generate_text(max_tokens=3)
        tokens = self.model._tokenize(text)

        self.assertLessEqual(len(tokens), 3)

    def test_generate_text_with_start_state(self):
        self.model.add_post("<START> Hello world <END>")

        text = self.model.generate_text(max_tokens=10, start_state=("<START>",))
        self.assertTrue(text.startswith("<START>"))

    def test_generate_text_empty_model(self):
        text = self.model.generate_text(max_tokens=10)
        self.assertEqual(text, "")

    def test_generate_text_stops_at_dead_end(self):
        self.model.add_post("Hello world")

        text = self.model.generate_text(max_tokens=100)
        tokens = self.model._tokenize(text)

        # Should be limited by available transitions
        self.assertGreater(len(tokens), 0)
        self.assertLessEqual(len(tokens), 3)  # "Hello", "world" and possibly some transitions


class TestParseBlogPostsFile(unittest.TestCase):

    def test_parse_with_reading_time(self):
        content = """================================================================================
POST 1: Test Post
Date: 2024-01-15
URL: https://example.com/test
================================================================================
Reading Time: 5 minutes This is the content of the post.


"""
        with patch('builtins.open', mock_open(read_data=content)):
            posts = parse_blog_posts_file('dummy.txt')

        self.assertEqual(len(posts), 1)
        self.assertEqual(posts[0], "This is the content of the post.")

    def test_parse_without_reading_time(self):
        content = """================================================================================
POST 1: Test Post
Date: 2024-01-15
URL: https://example.com/test
================================================================================
This is the content without reading time marker.


"""
        with patch('builtins.open', mock_open(read_data=content)):
            posts = parse_blog_posts_file('dummy.txt')

        self.assertEqual(len(posts), 1)
        self.assertIn("This is the content without reading time marker", posts[0])

    def test_parse_multiple_posts(self):
        content = """================================================================================
POST 1: First Post
Date: 2024-01-15
URL: https://example.com/post1
================================================================================
Reading Time: 5 minutes First post content.


================================================================================
POST 2: Second Post
Date: 2024-01-16
URL: https://example.com/post2
================================================================================
Reading Time: 3 minutes Second post content.


"""
        with patch('builtins.open', mock_open(read_data=content)):
            posts = parse_blog_posts_file('dummy.txt')

        self.assertEqual(len(posts), 2)
        self.assertEqual(posts[0], "First post content.")
        self.assertEqual(posts[1], "Second post content.")

    def test_parse_empty_file(self):
        content = ""
        with patch('builtins.open', mock_open(read_data=content)):
            posts = parse_blog_posts_file('dummy.txt')

        self.assertEqual(len(posts), 0)

    def test_parse_skips_empty_sections(self):
        content = """================================================================================
================================================================================
POST 1: Test Post
Date: 2024-01-15
URL: https://example.com/test
================================================================================
Reading Time: 5 minutes This is the content.


================================================================================
"""
        with patch('builtins.open', mock_open(read_data=content)):
            posts = parse_blog_posts_file('dummy.txt')

        self.assertEqual(len(posts), 1)
        self.assertEqual(posts[0], "This is the content.")


class TestBuildMarkovModelFromFile(unittest.TestCase):

    def test_build_model_from_file(self):
        content = """================================================================================
POST 1: Test Post
Date: 2024-01-15
URL: https://example.com/test
================================================================================
Reading Time: 5 minutes Hello world this is a test.


"""
        with patch('builtins.open', mock_open(read_data=content)):
            with patch('builtins.print'):  # Suppress print output
                model = build_markov_model_from_file('dummy.txt')

        self.assertIsInstance(model, MarkovModel)
        self.assertGreater(len(model.transitions), 0)

    def test_build_model_adds_special_tokens(self):
        content = """================================================================================
POST 1: Test Post
Date: 2024-01-15
URL: https://example.com/test
================================================================================
Reading Time: 5 minutes Hello world.


"""
        with patch('builtins.open', mock_open(read_data=content)):
            with patch('builtins.print'):
                model = build_markov_model_from_file('dummy.txt')

        # Check that <START> and <END> are in the model (as tuples for 1st order)
        self.assertIn(("<START>",), model.transitions)
        # <END> should appear in some token's transitions
        found_end = False
        for token_transitions in model.transitions.values():
            if "<END>" in token_transitions:
                found_end = True
                break
        self.assertTrue(found_end)

    def test_build_model_from_multiple_posts(self):
        content = """================================================================================
POST 1: First Post
Date: 2024-01-15
URL: https://example.com/post1
================================================================================
Reading Time: 5 minutes First post.


================================================================================
POST 2: Second Post
Date: 2024-01-16
URL: https://example.com/post2
================================================================================
Reading Time: 3 minutes Second post.


"""
        with patch('builtins.open', mock_open(read_data=content)):
            with patch('builtins.print'):
                model = build_markov_model_from_file('dummy.txt')

        # Should have 2 start states (one per post)
        self.assertEqual(len(model.start_states), 2)


if __name__ == '__main__':
    unittest.main()


class TestModelPersistence(unittest.TestCase):
    """Tests for saving and loading models."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.model = MarkovModel(order=2)
        self.model.add_post("Hello world this is a test")
        self.model.add_post("This is another test message")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.test_dir)

    def test_save_model(self):
        """Test that we can save a model to a file."""
        import tempfile
        model_file = os.path.join(self.test_dir, 'test_model.pkl')

        self.model.save(model_file)

        self.assertTrue(os.path.exists(model_file))
        self.assertGreater(os.path.getsize(model_file), 0)

    def test_load_model(self):
        """Test that we can load a saved model."""
        import tempfile
        model_file = os.path.join(self.test_dir, 'test_model.pkl')

        # Save
        self.model.save(model_file)

        # Load
        loaded_model = MarkovModel.load(model_file)

        # Verify it's the same
        self.assertEqual(loaded_model.order, self.model.order)
        self.assertEqual(len(loaded_model.transitions), len(self.model.transitions))
        self.assertEqual(len(loaded_model.start_states), len(self.model.start_states))

    def test_loaded_model_generates_text(self):
        """Test that a loaded model can generate text."""
        import tempfile
        model_file = os.path.join(self.test_dir, 'test_model.pkl')

        # Save
        self.model.save(model_file)

        # Load
        loaded_model = MarkovModel.load(model_file)

        # Generate text
        text = loaded_model.generate_text(max_tokens=20)

        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)

    def test_save_and_load_preserves_probabilities(self):
        """Test that probabilities are preserved after save/load."""
        import tempfile
        model_file = os.path.join(self.test_dir, 'test_model.pkl')

        # Get a probability before saving
        state = list(self.model.transitions.keys())[0]
        next_token = list(self.model.transitions[state].keys())[0]
        original_prob = self.model.get_transition_probability(state, next_token)

        # Save and load
        self.model.save(model_file)
        loaded_model = MarkovModel.load(model_file)

        # Check same probability
        loaded_prob = loaded_model.get_transition_probability(state, next_token)
        self.assertEqual(original_prob, loaded_prob)


class TestMarkovModelDunderMethods(unittest.TestCase):
    """Test dunder methods on MarkovModel."""

    def setUp(self):
        self.model = MarkovModel(order=1)
        self.model.add_post("<START> The quick brown fox jumps over the lazy dog <END>")

    def test_len_returns_number_of_states(self):
        """Test that __len__ returns number of unique states."""
        length = len(self.model)
        self.assertIsInstance(length, int)
        self.assertGreater(length, 0)
        self.assertEqual(length, len(self.model.transitions))

    def test_len_empty_model(self):
        """Test __len__ on empty model."""
        empty_model = MarkovModel(order=1)
        self.assertEqual(len(empty_model), 0)

    def test_len_changes_after_adding_posts(self):
        """Test that length increases after adding more posts."""
        initial_length = len(self.model)
        self.model.add_post("<START> A completely different set of words <END>")
        new_length = len(self.model)
        self.assertGreater(new_length, initial_length)

    def test_len_with_different_orders(self):
        """Test __len__ with different model orders."""
        model1 = MarkovModel(order=1)
        model1.add_post("<START> The quick brown fox <END>")
        len1 = len(model1)

        model2 = MarkovModel(order=2)
        model2.add_post("<START> The quick brown fox <END>")
        len2 = len(model2)

        # Different orders should have different number of states
        # (2nd order creates fewer states from same text)
        self.assertNotEqual(len1, len2)

    def test_getitem_with_tuple_state(self):
        """Test __getitem__ with tuple state."""
        state = ("the",)
        probs = self.model[state]
        self.assertIsInstance(probs, dict)
        self.assertGreater(len(probs), 0)
        # All probabilities should sum to 1
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=5)

    def test_getitem_with_string_state(self):
        """Test __getitem__ with string state (auto-converts to tuple)."""
        probs = self.model["the"]
        self.assertIsInstance(probs, dict)
        self.assertGreater(len(probs), 0)

    def test_getitem_string_equivalent_to_tuple(self):
        """Test that string and tuple access give same results."""
        probs1 = self.model["quick"]
        probs2 = self.model[("quick",)]
        self.assertEqual(probs1, probs2)

    def test_getitem_raises_keyerror_for_nonexistent_state(self):
        """Test that __getitem__ raises KeyError for nonexistent state."""
        with self.assertRaises(KeyError) as context:
            _ = self.model["nonexistent"]
        self.assertIn("nonexistent", str(context.exception))

    def test_getitem_with_second_order_model(self):
        """Test __getitem__ with 2nd-order model."""
        model2 = MarkovModel(order=2)
        model2.add_post("<START> The quick brown fox jumps <END>")

        # Access with 2-tuple
        probs = model2[("quick", "brown")]
        self.assertIsInstance(probs, dict)
        self.assertIn("fox", probs)

    def test_getitem_wrong_tuple_size_raises_error(self):
        """Test that accessing with wrong tuple size raises KeyError."""
        # 1st order model expects 1-tuples
        with self.assertRaises(KeyError):
            _ = self.model[("the", "quick")]

    def test_getitem_returns_correct_probabilities(self):
        """Test that __getitem__ returns correct probability values."""
        # Add controlled data
        model = MarkovModel(order=1)
        model.add_post("<START> cat cat cat dog <END>")

        probs = model["cat"]
        # "cat" appears 3 times, followed by "cat" twice and "dog" once
        # So: P(cat|cat) ≈ 0.667, P(dog|cat) ≈ 0.333
        self.assertIn("cat", probs)
        self.assertIn("dog", probs)
        self.assertAlmostEqual(probs["cat"], 2/3, places=2)
        self.assertAlmostEqual(probs["dog"], 1/3, places=2)

    def test_getitem_with_third_order_model(self):
        """Test __getitem__ with 3rd-order model."""
        model3 = MarkovModel(order=3)
        model3.add_post("<START> The quick brown fox jumps over the lazy dog <END>")

        # Access with 3-tuple
        probs = model3[("quick", "brown", "fox")]
        self.assertIsInstance(probs, dict)
        self.assertIn("jumps", probs)

    def test_getitem_empty_model_raises_keyerror(self):
        """Test that accessing empty model raises KeyError."""
        empty_model = MarkovModel(order=1)
        with self.assertRaises(KeyError):
            _ = empty_model["anything"]
