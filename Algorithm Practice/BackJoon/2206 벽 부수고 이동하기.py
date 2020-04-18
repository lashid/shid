rows, columns = map(int, input().split())
data = [list(map(int, input())) for repeat in range(rows)]

def search(x,y,z):
    for dx, dy in (0,1),(0,-1),(1,0),(-1,0):
        nx,ny=x+dx,y+dy
        if rows>nx>=0 and columns>ny>=0:
            if z:
                if not data[nx][ny] and not check[nx][ny][1]:
                    check[nx][ny][1]=1
                    candidates.append((nx,ny,z))
            else:
                if not data[nx][ny]:
                    if not check[nx][ny][0]:
                        check[nx][ny][0]=1
                        candidates.append((nx,ny,z))
                if data[nx][ny]:
                    candidates.append((nx,ny,1))

answers = []
check = [[[0,0] for repeat in range(columns)] for repeat in range(rows)]
candidates=[(0,0,0)]
check[0][0][0]=1
count=1
while candidates:
    for repeat in range(len(candidates)):
        current = candidates.pop(0)
        row, column, wall= current
        search(row,column,wall)
        if row==rows-1 and column==columns-1:
            answers.append(count)
    count+=1
    candidates=list(set(candidates))

if not answers:
    print(-1)
else:
    print(min(answers))