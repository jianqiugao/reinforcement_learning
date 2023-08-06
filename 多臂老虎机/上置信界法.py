import numpy as np

from 基础求解器 import solver
from 多臂老虎机 import BernoulliBandit, band_it_10_arm
from 画图 import plot_results


# 这个就是对未探索的地方加一个权重，让它去探索
class ucb(solver):
    def __init__(self, bandit: BernoulliBandit, coef, init_probs=1.):
        super(ucb, self).__init__(bandit)
        self.total_count = 0
        self.estimates = np.array([init_probs] * self.bandit.k)  # 得奖的概率，初始时刻均为1
        self.coef = coef

    def run_one_step(self):
        self.total_count += 1
        ucb = self.estimates + self.coef * np.sqrt(np.log(self.total_count) / (2 * (self.counts + 1))) # 主要计算一个ucb
        # 分数，这个分数会给探索较少的加权
        k = np.argmax(ucb)
        r = self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])  # 每一次选取后重新计算赢球的概率
        return k


ucb_solver = ucb(band_it_10_arm, coef=1)
ucb_solver.run(9000)
plot_results([ucb_solver], ['ucb_solver'])
