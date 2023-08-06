# 贪婪算法
import numpy as np

from 基础求解器 import solver
from 多臂老虎机 import BernoulliBandit, band_it_10_arm
from 画图 import plot_results


class EpsilonGreed(solver):
    def __init__(self, bandit: BernoulliBandit, epsilon=0.01, init_probs=1.):
        super(EpsilonGreed, self).__init__(bandit)
        self.epsilon = epsilon
        self.estimates = np.array([init_probs] * self.bandit.k)  # 得奖的概率，初始时刻均为1

    def run_one_step(self):
        if np.random.random() < self.epsilon:  # 当小于这个概率的时候
            k = np.random.randint(0, self.bandit.k)  # 随机选择一个机器
        else:
            k = np.argmax(self.estimates)  # 否则就选取最大概率的机器
        r = self.bandit.step(k)  # 返回是赢还是输，1是赢
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])  # 每一次选取后重新计算赢球的概率
        return k


np.random.seed(1)


# greed_solver = EpsilonGreed(band_it_10_arm, epsilon=0.01)
# greed_solver.run(5000)
# plot_results([greed_solver], ['greed_solver'])
# 后悔值累计的越高越有问题，如果后悔值累计到0不懂那是最好的

# epsilons = [1e-4, 0.01, 0.1, 0.25, 0.5]
# epsilons_greed_solver_list = [EpsilonGreed(band_it_10_arm, epsilon=e) for e in epsilons]
# epsilons_solver_name_list = [f"epsilon{e}" for e in epsilons]
# for solver in epsilons_greed_solver_list:
#     solver.run(5000)
# plot_results(epsilons_greed_solver_list, epsilons_solver_name_list)


# 随着迭代步的进行逐渐衰减的方案

class DecayingEpsilonGreedy(solver):
    def __init__(self, bandit: BernoulliBandit, init_probs=1.):
        super(DecayingEpsilonGreedy, self).__init__(bandit)
        self.estimates = np.array([init_probs] * self.bandit.k)  # 得奖的概率，初始时刻均为1
        self.total_count = 0

    def run_one_step(self):
        self.total_count += 1
        if np.random.random() < 1 / self.total_count:
            k = np.random.randint(0, self.bandit.k)
        else:
            k = np.argmax(self.estimates)  # 否则就选取最大概率的机器

        r = self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])  # 每一次选取后重新计算赢球的概率
        return k


decay_greed_solver = DecayingEpsilonGreedy(band_it_10_arm)
decay_greed_solver.run(5000)
plot_results([decay_greed_solver],['decay_greed_solver'])
