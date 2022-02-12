import numpy as np

def apk(actual, predicted, k=7, default=0.0):

    # MAP@7 이므로, 최대 7개만 사용
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    # 예측값이 정답에 있고
    # 예측값이 중복이 아니면
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)


    if not actual:
        return default

    return score / min(len(actual), k)

def mapk(actual, predicted, k=7, default=0.0):
    return np.mean([apk(a, p, k, default) for a, a in zip(actual, predicted)])

