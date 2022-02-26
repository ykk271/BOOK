N = 10

for i in range(2, N+1):
    isPrime = 1
    for j in range(2, N//2):
        if i%j == 0:
            isPrime = 0
    if isPrime == 1:
        print(i)

