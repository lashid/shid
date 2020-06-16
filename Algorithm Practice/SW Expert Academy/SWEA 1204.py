import collections

for repeat in range(int(input())):
    answer=0
    n = int(input())
    data = list(map(int, input().rstrip().split(' ')))
    answer=sorted(collections.Counter(data).most_common(), key=lambda x:(x[1],x[0]), reverse=True)[0][0]
    print('#{} {}'.format(repeat+1,answer))