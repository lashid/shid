size = int(input())
data = [list(map(int, input())) for repetition in range(size)]
count = int()
check = [[False]*size for repetition in range(size)]

house_count = 1
house_counts = []

def search(x,y):
    check[x][y] = True
    for dx, dy in (1,0), (-1,0), (0,1), (0,-1):
        nx, ny = x+dx, y+dy
        if size > nx >= 0 and size > ny >= 0 and data[nx][ny] and not check[nx][ny]:
            search(nx,ny)
            global house_count
            house_count += 1

for x in range(size):
    for y in range(size):
        if data[x][y] and not check[x][y]:
            search(x,y)
            count += 1
            house_counts.append(house_count)
            house_count = 1

house_counts.sort()

print(count)
for house_count in house_counts:
    print(house_count)