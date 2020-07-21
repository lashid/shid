for test_case in range(1, int(input())+1):
    N = int(input())
    data = [str(input()) for repetition in range(N)]
    data = sorted(list(set(data)), key=lambda x:[len(x),x])
    print(f"#{test_case}")
    for name in data:
        print(name)