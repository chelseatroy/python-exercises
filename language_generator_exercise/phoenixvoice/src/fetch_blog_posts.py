import requests
import time

def query_wordpress_api(base_url="https://chelseatroy.com", start_page=1, end_page=None, max_posts=None):
    """
    Fetch all blog posts from a WordPress site using the REST API.

    Args:
        base_url: The base URL of the WordPress site
        start_page: Page number to start from (default: 1)
        end_page: Page number to end at (default: None, fetch all)
        max_posts: Maximum number of posts to fetch (default: None, fetch all)

    Returns:
        List of post dictionaries containing post data
    """
    posts = []
    page = start_page
    per_page = 100  # Maximum allowed by WordPress API

    while True:
        # WordPress REST API endpoint for posts
        url = f"{base_url}/wp-json/wp/v2/posts"
        params = {
            "page": page,
            "per_page": per_page,
            "_embed": True  # Include embedded data like featured images
        }

        print(f"Fetching page {page}...")

        try:
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()

            page_posts = response.json()

            # If no posts returned, we've reached the end
            if not page_posts:
                break

            posts.extend(page_posts)
            print(f"  Retrieved {len(page_posts)} posts (total so far: {len(posts)})")

            # Check if we've reached the max_posts limit
            if max_posts and len(posts) >= max_posts:
                posts = posts[:max_posts]  # Trim to exact number requested
                print(f"Reached maximum post limit of {max_posts}")
                break

            # Check if there are more pages
            total_pages = response.headers.get('X-WP-TotalPages')
            if total_pages and page >= int(total_pages):
                break

            # Check if we've reached the end page
            if end_page and page >= end_page:
                break

            page += 1

            # Be nice to the server
            time.sleep(0.5)

        except requests.exceptions.RequestException as e:
            print(f"Error fetching page {page}: {e}")
            break

    return posts


def extract_text_from_posts(posts):
    """
    Extract title and content text from posts.

    Args:
        posts: List of post dictionaries from WordPress API

    Returns:
        Formatted string containing all post text
    """
    output_lines = []

    for i, post in enumerate(posts, 1):
        title = post.get('title', {}).get('rendered', 'Untitled')
        content = post.get('content', {}).get('rendered', '')
        date = post.get('date', 'Unknown date')
        link = post.get('link', '')

        # Remove HTML tags from content (basic approach)
        import re
        content_text = re.sub(r'<[^>]+>', '', content)
        content_text = re.sub(r'\s+', ' ', content_text).strip()

        # Remove HTML entities from title
        title_text = re.sub(r'<[^>]+>', '', title)

        output_lines.append(f"{'='*80}")
        output_lines.append(f"POST {i}: {title_text}")
        output_lines.append(f"Date: {date}")
        output_lines.append(f"URL: {link}")
        output_lines.append(f"{'='*80}")
        output_lines.append(content_text)
        output_lines.append("\n")

    return "\n".join(output_lines)


def fetch_blog_posts(base_url="https://chelseatroy.com", max_posts=None, start_page=1, end_page=None, append=False):
    """
    Main function to fetch posts and save to file.

    Returns:
        str: Path to the output file containing the blog posts
    """
    print(f"Starting to fetch blog posts from {base_url}...")
    if max_posts:
        print(f"Maximum posts to fetch: {max_posts}")
    print(f"Fetching pages {start_page} to {end_page if end_page else 'end'}")
    print()

    # Fetch all posts
    posts = query_wordpress_api(base_url, start_page=start_page, end_page=end_page, max_posts=max_posts)

    if not posts:
        print("No posts were retrieved.")
        return None

    print()
    print(f"Successfully retrieved {len(posts)} posts!")
    print()

    # Extract text
    print("Extracting text from posts...")
    all_text = extract_text_from_posts(posts)

    # Save to file
    # Generate filename from URL
    from urllib.parse import urlparse
    domain = urlparse(base_url).netloc.replace('www.', '').replace('.', '_')
    output_file = f"{domain}_blog_posts.txt"

    mode = 'a' if append else 'w'
    with open(output_file, mode, encoding='utf-8') as f:
        f.write(all_text)

    action = "appended to" if append else "saved to"
    print(f"Blog post text {action} {output_file}")
    print(f"Total characters in this batch: {len(all_text):,}")

    return output_file
