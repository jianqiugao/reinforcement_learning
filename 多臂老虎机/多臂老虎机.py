# 实现一个拉杆为10的多臂老虎机
import numpy as np
import matplotlib.pyplot as plt


class BernoulliBandit:
    def __init__(self, k):
        self.probs = np.random.uniform(size=k)
        self.best_idx = np.argmax(self.probs)
        self.best_prob = self.probs[self.best_idx]
        self.k = k

    def step(self, k):
        if np.random.rand() < self.probs[k]:
            return 1
        else:
            return 0
np.random.seed(1)

k = 10
band_it_10_arm = BernoulliBandit(10)
if __name__ == '__main__':
    print(f"随机生成了一个{k}臂老虎机")
    print(f"最大概率的杆为{band_it_10_arm.best_idx}号,最大概率是{band_it_10_arm.probs[band_it_10_arm.best_idx]}")
