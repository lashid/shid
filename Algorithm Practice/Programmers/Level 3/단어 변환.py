def solution(begin, target, words):
    answer = []
    words.append(begin)
    length = len(words)
    links = [[] for idx in range(length)]

    def check(x, y):
        count = 0
        for idx in range(len(x)):
            if x[idx] != y[idx]:
                count += 1
            if count >= 2:
                return 0
                break
        if count == 1:
            return 1

    for x in range(length):
        for y in range(length):
            if check(words[x], words[y]):
                links[x].append(y)

    def search(x,y,z):
        z[x]=True
        for link in links[x]:
            if link==length-1:
                continue
            if link==words.index(target):
                answer.append(y+2)
                break
            elif not z[link]:
                search(link,y+1,z.copy())

    starts = links.pop()

    if target in words:
        if words.index(target) in starts:
            answer.append(1)
        for start in starts:
            checks = [False] * (length - 1)
            search(start,0,checks.copy())
    else:
        answer=[0]

    answer=min(answer)
    return answer