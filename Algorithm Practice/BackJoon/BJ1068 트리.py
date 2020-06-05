import sys
sys.setrecursionlimit(1000000)

N=int(input())
nodes=list(map(int, input().split(' ')))
target=int(input())
answer=N

check=[False]*N
answer-=1

def search(x):
    global answer
    check[x] = True
    for index in range(N):
        if nodes[index]==x and not check[index]:
            answer-=1
            search(index)

search(target)
temp=[nodes[index] for index in range(N) if not check[index]]
answer-=len(set(temp))-1

if nodes[target]==-1:
    answer=0

print(answer)