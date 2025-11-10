#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# Add phoenixvoice to path
sys.path.insert(0, str(Path(__file__).parent / "phoenixvoice"))
from phoenixvoice.src.markov_voice_generator import MarkovVoiceGenerator
from phoenixvoice.src.fetch_blog_posts import fetch_blog_posts

def main():
    # Configuration
    blog_url = "https://chelseatroy.com"
    blog_file = "chelseatroy_blog_posts.txt"
    max_posts = 50  

    # Step 1: Get or fetch blog posts
    if not os.path.exists(blog_file):
        fetch_blog_posts(base_url=blog_url, max_posts=max_posts)

    # Step 2: Train Markov model
    markov_generator = MarkovVoiceGenerator(order=4)
    markov_generator.train(blog_file)

    # Step 3: Generate text with Markov model
    markov_text = markov_generator.generate_text(max_tokens=300)
    print(markov_text)

if __name__ == "__main__":
    main()
