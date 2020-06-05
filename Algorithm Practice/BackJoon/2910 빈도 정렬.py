N,C=map(int, input().split(' '))
data=list(map(int, input().split(' ')))

import collections

data=collections.Counter(data).most_common()

answer=[str(number) for number,repeat in data for _ in range(repeat)]

print(' '.join(answer))
