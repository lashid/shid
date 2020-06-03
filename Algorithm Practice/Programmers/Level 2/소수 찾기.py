import itertools

def filtering(x):
    return x>1

def solution(numbers):
    length=len(numbers)
    candidates=list()
    for index in range(1,length+1):
        candidates.extend(list(map(int, list(map(''.join, itertools.permutations(numbers, index))))))
    candidates=list(filter(filtering, list(set(candidates))))

    def check(x, counts):
        for number in range(2, x):
            if x % number == 0:
                counts -= 1
                break
        return counts

    counts=len(candidates)

    for candidate in candidates:
        counts=check(candidate,counts)

    return counts