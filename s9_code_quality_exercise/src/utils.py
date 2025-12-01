WORD_TO_NUMBER = {
    'zero': 0, '0': 0,
    'one': 1, '1': 1,
    'two': 2, '2': 2,
    'three': 3, '3': 3,
    'four': 4, '4': 4,
    'five': 5, '5': 5,
    'six': 6, '6': 6,
    'seven': 7, '7': 7,
    'eight': 8, '8': 8,
    'nine': 9, '9': 9,
    'ten': 10, '10': 10
}

def get_num(word):
    word = word.strip().lower()

    if word in WORD_TO_NUMBER:
        return WORD_TO_NUMBER[word]

    try:
        return int(word)
    except:
        raise ValueError("Unknown number")
