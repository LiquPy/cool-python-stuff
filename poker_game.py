# Python 3 
# initally based on Udacity's CS 212 - Lesson 1 (https://classroom.udacity.com/courses/cs212/lessons/48688918/concepts/486865220923)
import random

num_of_decks = 2
mydeck = [r+s for r in '23456789TJQKA' for s in 'SHDC']*num_of_decks

def deal(numhands, n=5, deck=mydeck):
    random.shuffle(deck)
    return [deck[n*i:n*(i+1)] for i in range(numhands)]

def poker(hands):
    """
    input a set of hands, return winner hand
    """
    return allmax(hands, key=hand_rank)

def allmax(iterable, key=None):
    result, maxval = [], None
    key = key or (lambda x: x)
    for x in iterable:
        xval = key(x)
        print(x, xval)
        if result == [] or xval > maxval:
            result, maxval = [x], xval
        elif xval == maxval:
            result.append(x)
    return result

def hand_rank(hand):
    ranks = card_ranks(hand)
    if straight(ranks) and flush(hand):
        return (8, max(ranks))
    elif kind(4, ranks):
        return (7, kind(4, ranks), kind(1, ranks))
    elif kind(3, ranks) and kind(2, ranks):
        return (6, kind(3, ranks), kind(2, ranks))
    elif flush(hand):
        return (5, ranks)
    elif straight(ranks):
        return (4, max(ranks))
    elif kind(3, ranks):
        return (3, kind(3, ranks), ranks)
    elif two_pair(ranks):
        return (2, two_pair(ranks), ranks)
    elif kind(2, ranks):
        return (1, kind(2, ranks), ranks)
    else:
        return (0, ranks)

def card_ranks(cards):
    ranks = ['--23456789TJQKA'.index(r) for r,s in cards]
    ranks.sort(reverse=True)
    return [5, 4, 3, 2, 1] if (ranks == [14, 5, 4, 3, 2]) else ranks

def straight(ranks):
    return (max(ranks)-min(ranks) == 4) and len(set(ranks)) == 5

def flush(hand):
    return len(set([s for r,s in hand])) == 1
    
def kind(n, ranks):
    for r in ranks:
        if ranks.count(r) == n: return r
    return None

def two_pair(ranks):
    pair = kind(2, ranks)
    lowpair = kind(2, list(reversed(ranks)))
    if pair and lowpair != pair:
        return (pair, lowpair)
    return None

hands = deal(7)
print(poker(hands))
