import sys
sys.setrecursionlimit(100000)

rows, columns=map(int, input().split(" "))
data=[]
for repeat in range(rows):
    data.append(list(map(int, input().rstrip().split(' '))))

def melt(x,y):
    for dx, dy in (0,1),(0,-1),(1,0),(-1,0):
        if not data[x+dx][y+dy]:
            add[x][y]+=1

def check(x,y):
    checklist[x][y]=True
    for dx, dy in (0,1),(0,-1),(1,0),(-1,0):
        if data[x+dx][y+dy] and not checklist[x+dx][y+dy]:
            check(x+dx,y+dy)
answer=0
while True:
    add = [[0] * columns for repeat in range(rows)]
    for x in range(1, rows-1):
        for y in range(1, columns-1):
            if data[x][y]:
                melt(x,y)
    for x in range(1, rows-1):
        for y in range(1, columns-1):
            melt_value=add[x][y]
            value=data[x][y]
            if melt_value:
                if melt_value>=value:
                    data[x][y]=0
                else:
                    data[x][y]=value-melt_value
    answer+=1
    count=0
    checklist = [[False] * columns for repeat in range(rows)]
    for x in range(1, rows-1):
        for y in range(1, columns-1):
            if data[x][y] and not checklist[x][y]:
                check(x,y)
                count+=1
    if count>=2:
        break
    if count==0:
        answer=0
        break

print(answer)