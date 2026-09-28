"""Utility functions for number word conversion."""

# Number word dictionary - SINGLE SOURCE OF TRUTH for word-to-number mapping
WORD_TO_NUMBER = {
    'zero': 0,
    'one': 1,
    'two': 2,
    'three': 3,
    'four': 4,
    'five': 5,
    'six': 6,
    'seven': 7,
    'eight': 8,
    'nine': 9,
    'ten': 10,
}


def get_num(word):
    """
    Convert a word or digit string to a number.

    Args:
        word: Number word (e.g., "five") or digit string (e.g., "5")

    Returns:
        The numeric value

    Raises:
        ValueError: If word is not a recognized number word or valid digit string
    """
    word = word.strip().lower()

    # Check if it's a word number
    if word in WORD_TO_NUMBER:
        return WORD_TO_NUMBER[word]

    # Try to parse as numeric string
    try:
        return int(word)
    except ValueError:
        raise ValueError(f"Unknown number word or invalid number: '{word}'")
