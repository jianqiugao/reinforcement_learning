import numpy as np

from 基础求解器 import solver
from 多臂老虎机 import BernoulliBandit, band_it_10_arm
from 画图 import plot_results


class ThompsonSampling(solver):
    def __init__(self, bandit: BernoulliBandit):
        super(ThompsonSampling, self).__init__(bandit)
        self._a = np.ones(self.bandit.k)  # 每个拉杆奖励为1的个数
        self._b = np.ones(self.bandit.k)  # 每个拉杆奖励为1的个数

    def run_one_step(self):
        sample = np.random.beta(self._a, self._b)  # 取这一对值得概率
        k = np.argmax(sample)
        r = self.bandit.step(k)
        self._a[k] += r
        self._b[k] += (1 - r)
        return k


ThompsonSampling_solver = ThompsonSampling(band_it_10_arm)
ThompsonSampling_solver.run(9000)
plot_results([ThompsonSampling_solver], ['ThompsonSampling_solver'])
