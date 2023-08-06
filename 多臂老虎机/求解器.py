import matplotlib.pyplot as plt
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
        self.regret += self.bandit.best_prob - self.bandit.probs[k]
        self.regrets.append(self.regret)

    def run_one_step(self):
        raise NotImplementedError

    def run(self, num_steps):
        for _ in range(num_steps):
            k = self.run_one_step()
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k)


def plot_results(solvers, solver_name):
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_name[idx])
    plt.xlabel('time_steps')
    plt.ylabel('cumulative regret ')
    plt.legend()
    plt.show()


# 贪婪算法

class EpsilonGreed(solver):
    def __init__(self, bandit: BernoulliBandit, epsilon=0.01, init_probs=1):
        super(EpsilonGreed, self).__init__(bandit)
        self.epsilon = epsilon
        self.estimates = np.array([init_probs] * self.bandit.k)

    def run_one_step(self):
        if np.random.random() < self.epsilon:
            k = np.random.randint(0, self.bandit.k)
        else:
            k = np.argmax(self.estimates)
        r = self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        return k


np.random.seed(1)
greed_solver = EpsilonGreed(band_it_10_arm, epsilon=0.01)
greed_solver.run(5000)
plot_results([greed_solver], ['greed_solver'])
