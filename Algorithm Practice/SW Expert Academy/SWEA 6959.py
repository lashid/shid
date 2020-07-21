for test_case in range(1, int(input())+1):
    number = list(map(int, input()))
    sum = -1
    for num in number:
        sum += num
    if len(number) == 1 or (len(number) - 1 + (sum//9))%2 == 0:
        print(f"#{test_case} B")
    else:
        print(f"#{test_case} A")