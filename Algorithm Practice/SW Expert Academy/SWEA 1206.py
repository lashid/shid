for repeat in range(10):
    answer=0
    n = int(input())
    data = list(map(int, input().rstrip().split(' ')))
    a,b,c,d,e=0,0,data[2],data[3],data[4]
    index=4
    while index!=n:
        e=data[index]
        max_num=max([a, b, c, d, e])
        max_others=max([a,b,d,e])
        if max_num==c:
            answer+=(c-max_others)
        index += 1
        a,b,c,d=b,c,d,e
    print('#{} {}'.format(repeat+1,answer))