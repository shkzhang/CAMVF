import numpy as np
from itertools import product


def sim_D(n):
    def hamming_distance(s1, s2):
        return sum(c1 != c2 for c1, c2 in zip(s1, s2))

    def jaccard_similarity(s1, s2):
        union = len(set(s1).union(set(s2)))
        intersection = len(set(s1).intersection(set(s2)))
        return intersection / union if union != 0 else 0

    # 初始化矩阵
    matrix = np.zeros((2 ** n, 2 ** n))
    # 得到集合的所有子集
    subsets = list(product([0, 1], repeat=n))

    # 根据集合相似度计算矩阵
    for i in range(2 ** n):
        for j in range(i, 2 ** n):
            subset1 = subsets[i]
            subset2 = subsets[j]
            similarity = jaccard_similarity(subset1, subset2)
            matrix[i, j] = similarity
            matrix[j, i] = similarity

    return matrix


# 测试示例
n = 1
result = sim_D(n)
print(result)
