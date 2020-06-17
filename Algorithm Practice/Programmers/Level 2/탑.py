def solution(heights):
    length = len(heights)
    answer = [0 for repeat in range(length)]

    def search(x):
        height = heights[x]
        for i in range(x - 1, -1, -1):
            if heights[i] > height:
                return i + 1
                break
        return 0

    for i in range(length):
        answer[i] = search(i)

    return answer