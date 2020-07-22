for test_case in range(1, int(input())+1):
    size = int(input())
    incomes = list(map(int, input().split()))
    mean_incomes = sum(incomes) / size
    answer = 0

    for income in incomes:
        answer += 1 if income <= mean_incomes else 0

    print(f"#{test_case} {answer}")