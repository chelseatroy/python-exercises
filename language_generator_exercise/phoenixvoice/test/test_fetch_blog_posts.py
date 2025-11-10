import unittest
from unittest.mock import Mock, patch, mock_open
import json
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phoenixvoice.src.fetch_blog_posts import fetch_all_posts, extract_text_from_posts, fetch_blog_posts


class TestFetchBlogPosts(unittest.TestCase):

    def setUp(self):
        self.sample_post = {
            'title': {'rendered': 'Test Post Title'},
            'content': {'rendered': '<p>This is test content with <strong>HTML</strong> tags.</p>'},
            'date': '2024-01-15T10:30:00',
            'link': 'https://example.com/test-post'
        }

    @patch('phoenixvoice.src.fetch_blog_posts.requests.get')
    def test_fetch_all_posts_single_page(self, mock_get):
        mock_response = Mock()
        mock_response.json.return_value = [self.sample_post]
        mock_response.headers.get.return_value = '1'
        mock_get.return_value = mock_response

        posts = fetch_all_posts("https://example.com", start_page=1, end_page=None)

        self.assertEqual(len(posts), 1)
        self.assertEqual(posts[0]['title']['rendered'], 'Test Post Title')
        mock_get.assert_called_once()

    @patch('phoenixvoice.src.fetch_blog_posts.requests.get')
    def test_fetch_all_posts_multiple_pages(self, mock_get):
        mock_response_page1 = Mock()
        mock_response_page1.json.return_value = [self.sample_post] * 100
        mock_response_page1.headers.get.return_value = '2'

        mock_response_page2 = Mock()
        mock_response_page2.json.return_value = [self.sample_post] * 50
        mock_response_page2.headers.get.return_value = '2'

        mock_get.side_effect = [mock_response_page1, mock_response_page2]

        posts = fetch_all_posts("https://example.com")

        self.assertEqual(len(posts), 150)
        self.assertEqual(mock_get.call_count, 2)

    @patch('phoenixvoice.src.fetch_blog_posts.requests.get')
    def test_fetch_all_posts_with_max_posts(self, mock_get):
        mock_response = Mock()
        mock_response.json.return_value = [self.sample_post] * 100
        mock_response.headers.get.return_value = '5'
        mock_get.return_value = mock_response

        posts = fetch_all_posts("https://example.com", max_posts=50)

        self.assertEqual(len(posts), 50)

    @patch('phoenixvoice.src.fetch_blog_posts.requests.get')
    def test_fetch_all_posts_handles_empty_response(self, mock_get):
        mock_response = Mock()
        mock_response.json.return_value = []
        mock_get.return_value = mock_response

        posts = fetch_all_posts("https://example.com")

        self.assertEqual(len(posts), 0)

    @patch('phoenixvoice.src.fetch_blog_posts.requests.get')
    @patch('phoenixvoice.src.fetch_blog_posts.time.sleep')
    def test_fetch_all_posts_handles_request_exception(self, mock_sleep, mock_get):
        import requests
        mock_get.side_effect = requests.exceptions.RequestException("Network error")

        posts = fetch_all_posts("https://example.com")

        self.assertEqual(len(posts), 0)

    @patch('phoenixvoice.src.fetch_blog_posts.requests.get')
    def test_fetch_all_posts_respects_end_page(self, mock_get):
        mock_response = Mock()
        mock_response.json.return_value = [self.sample_post] * 100
        mock_response.headers.get.return_value = '10'
        mock_get.return_value = mock_response

        posts = fetch_all_posts("https://example.com", start_page=1, end_page=2)

        self.assertEqual(len(posts), 200)
        self.assertEqual(mock_get.call_count, 2)

    def test_extract_text_from_posts_single_post(self):
        posts = [self.sample_post]
        text = extract_text_from_posts(posts)

        self.assertIn('Test Post Title', text)
        self.assertIn('This is test content with HTML tags', text)
        self.assertIn('https://example.com/test-post', text)
        self.assertNotIn('<p>', text)
        self.assertNotIn('<strong>', text)

    def test_extract_text_from_posts_multiple_posts(self):
        posts = [self.sample_post, self.sample_post]
        text = extract_text_from_posts(posts)

        self.assertIn('POST 1:', text)
        self.assertIn('POST 2:', text)

    def test_extract_text_from_posts_strips_html(self):
        post_with_html = {
            'title': {'rendered': '<em>Title with HTML</em>'},
            'content': {'rendered': '<div><p>Content with <a href="#">link</a></p></div>'},
            'date': '2024-01-15T10:30:00',
            'link': 'https://example.com/post'
        }

        text = extract_text_from_posts([post_with_html])

        self.assertNotIn('<em>', text)
        self.assertNotIn('<div>', text)
        self.assertNotIn('<a href', text)
        self.assertIn('Title with HTML', text)
        self.assertIn('Content with link', text)

    def test_extract_text_from_posts_handles_missing_fields(self):
        minimal_post = {
            'title': {},
            'content': {}
        }

        text = extract_text_from_posts([minimal_post])

        self.assertIn('Untitled', text)
        self.assertIn('Unknown date', text)

    @patch('phoenixvoice.src.fetch_blog_posts.fetch_all_posts')
    @patch('builtins.open', new_callable=mock_open)
    def test_main_creates_file_with_correct_name(self, mock_file, mock_fetch):
        mock_fetch.return_value = [self.sample_post]

        fetch_blog_posts(base_url="https://example.com")

        mock_file.assert_called_once_with('example_com_blog_posts.txt', 'w', encoding='utf-8')

    @patch('phoenixvoice.src.fetch_blog_posts.fetch_all_posts')
    @patch('builtins.open', new_callable=mock_open)
    def test_main_appends_when_flag_set(self, mock_file, mock_fetch):
        mock_fetch.return_value = [self.sample_post]

        fetch_blog_posts(base_url="https://example.com", append=True)

        mock_file.assert_called_once_with('example_com_blog_posts.txt', 'a', encoding='utf-8')

    @patch('phoenixvoice.src.fetch_blog_posts.fetch_all_posts')
    def test_main_handles_no_posts(self, mock_fetch):
        mock_fetch.return_value = []

        # Should not raise an exception
        fetch_blog_posts(base_url="https://example.com")

    @patch('phoenixvoice.src.fetch_blog_posts.fetch_all_posts')
    @patch('builtins.open', new_callable=mock_open)
    def test_main_passes_max_posts_to_fetch(self, mock_file, mock_fetch):
        mock_fetch.return_value = [self.sample_post]

        fetch_blog_posts(base_url="https://example.com", max_posts=10)

        mock_fetch.assert_called_once_with("https://example.com", start_page=1, end_page=None, max_posts=10)


if __name__ == '__main__':
    unittest.main()
