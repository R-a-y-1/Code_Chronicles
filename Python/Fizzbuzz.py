def fizzbuzz():                       # Creating a function to play the game Fizzbuzz. Not concerned with optimisation just yet.
    x = list(range(1, 101, 1))
    for x in x:
        i = ''
        if x % 3 == 0:
            i += 'fizz'
        if x % 5 == 0:
            i += 'buzz'
        else:
            i += str(x)
    return i

if __name__ == "__main__":
    fizzbuzz()