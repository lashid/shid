for repeat in range(int(input())):
    x,y=map(int, input().split(' '))
    candidates=list(map(int, input().split(' ')))
    data=[i for i in range(x)]
    count=0
    def solution(x):
        global data, count
        if x[0] != max(x):
            x.append(x.pop(0))
            data.append(data.pop(0))
        else:
            x.pop(0)
            count+=1
            return data.pop(0)
    while True:
        if solution(candidates)==y:
            break
    print(count)