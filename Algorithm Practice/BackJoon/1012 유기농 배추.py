import sys
sys.setrecursionlimit(100000)

def search(x,y):
    check[y][x] = True
    for dx, dy in (1,0), (-1,0), (0,1), (0,-1):
        nx, ny = x+dx, y+dy
        if columns > nx >= 0 and rows > ny >= 0 and data[ny][nx] and not check[ny][nx]:
            search(nx,ny)

for repetition in range(int(input())):
    columns, rows, locations = map(int, input().split())
    data = [[False]*columns for repetition in range(rows)]
    answer = 0

    for location in range(locations):
        x, y = map(int, input().split())
        data[y][x] = True

    check = [[False]*columns for repetition in range(rows)]

    for x in range(columns):
        for y in range(rows):
            if not check[y][x] and data[y][x]:
                search(x,y)
                answer += 1

    print(answer)