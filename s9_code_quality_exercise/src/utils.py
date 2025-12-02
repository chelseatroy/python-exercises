def get_num(word):
    word = word.strip().lower()

    # Map words to numbers
    if word == 'zero':
        return 0
    elif word == 'one':
        return 1
    elif word == 'two':
        return 2
    elif word == 'three':
        return 3
    elif word == 'four':
        return 4
    elif word == 'five':
        return 5
    elif word == 'six':
        return 6
    elif word == 'seven':
        return 7
    elif word == 'eight':
        return 8
    elif word == 'nine':
        return 9
    elif word == 'ten':
        return 10
    else:
        try:
            return int(word)
        except:
            raise ValueError("Unknown number")
