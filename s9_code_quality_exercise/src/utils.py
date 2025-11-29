def get_num(word):
    word = word.strip().lower()

    # Map words to numbers
    if word == 'zero' or word == '0':
        return 0
    elif word == 'one' or word == '1':
        return 1
    elif word == 'two' or word == '2':
        return 2
    elif word == 'three' or word == '3':
        return 3
    elif word == 'four' or word == '4':
        return 4
    elif word == 'five' or word == '5':
        return 5
    elif word == 'six' or word == '6':
        return 6
    elif word == 'seven' or word == '7':
        return 7
    elif word == 'eight' or word == '8':
        return 8
    elif word == 'nine' or word == '9':
        return 9
    elif word == 'ten' or word == '10':
        return 10
    else:
        try:
            return int(word)
        except:
            raise ValueError("Unknown number")
