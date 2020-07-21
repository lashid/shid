def up(x):
    length = len(x)
    for a in range(length):
        before = [0,0,0]
        for b in range(length):
            current = x[b][a]
            if current:
                if not before[0]:
                    before = [current,b,a]
                elif before[0] == current:
                    x[before[1]][before[2]] = current*2
                    x[b][a] = 0
                    before = [0,0,0]
                else:
                    before = [current,b,a]

    for a in range(length):
        for b in range(length):
            if not x[b][a]:
                for c in range(b, length-1):
                    if x[c+1][a]:
                        x[b][a] = x[c+1][a]
                        x[c+1][a] = 0
                        break
    return x

for test_case in range(1, int(input())+1):
    size, move = input().split()
    data = [list(map(int, input().split())) for repetition in range(int(size))]
    if move == 'up':
	    print(up(data))