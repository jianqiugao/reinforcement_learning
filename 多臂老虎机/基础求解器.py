
import numpy as np

from 多臂老虎机 import band_it_10_arm, BernoulliBandit


class solver:
    def __init__(self, bandit: BernoulliBandit):
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.k)
        self.regret = 0  # 当前步的累计懊悔
        self.actions = []
        self.regrets = []

    def update_regret(self, k):
        self.regret += self.bandit.best_prob - self.bandit.probs[k]  # 当前步的(最优的概率-当前概率)，这个累计的越低代表赢的次数比较多
        self.regrets.append(self.regret)

    def run_one_step(self):
        raise NotImplementedError

    def run(self, num_steps):
        for _ in range(num_steps):
            k = self.run_one_step()
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k)

