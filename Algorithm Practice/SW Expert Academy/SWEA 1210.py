def search_rows(x,y):
    global check
    check[x][y] = True
    for dy in 1, -1:
        ny = y+dy
        if 99 >= ny >= 0 and data[x][ny] and not check[x][ny]:
            return x, y
    nx = x-1
    if x != 0:
        return search_rows(nx,y)
    else:
        global answer
        answer = y
        return x, y

def search_columns(x,y):
    global check
    check[x][y] = True
    for dy in 1, -1:
        ny = y+dy
        if 99 >= ny >= 0 and data[x][ny] and not check[x][ny]:
            while 99 >= ny >= 0 and data[x][ny]:
                check[x][ny] = True
                ny += dy
            return x, ny-dy

for test_case in range(1,11):
    n = int(input())
    data = [list(map(int, input().rstrip().split())) for repetition in range(100)]
    check = [[False]*100 for repetition in range(100)]
    target = data[-1].index(2)
    answer = -1
    a, b = 99, target

    while True:
        a, b = search_rows(a,b)
        if a == 0:
            break
        a, b = search_columns(a,b)

    print(answer)