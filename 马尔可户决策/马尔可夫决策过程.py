import numpy as np
# 状态价值函数

# 动作价值函数

# 贝尔曼期望

# 状态集合
s = ["s1", "s2", "s3", "s4", "s5"]
# 动作集合
a = ["保持s1", "前往s1", "前往s2", "前往s2", "前往s3", "前往s4","前往s5","概率前往"]
# 状态转移函数
p = {"s1-保持s1-s1"}  # 。。。。。

# 奖励函数
r = {}

gamma = 0.5 # 折扣因子
MDP = (s,a,p,r,gamma)  # 马尔科夫决策过程
MRP = [] # 马尔科夫奖励过程

# 策略1 随机策略
p_from_mdp_to_mrp = [[0.5,0.5,0,0,0],
                     [0.5,0,0.5,0,0],
                     [0,0,0,0.5,0.5],
                     [0,0.1,0.2,0.2,0.5],
                     [0.0,0.0,0.0,0.0,1]
                     ]
# 分析策略的状态价值函数
p_from_mdp_to_mrp = np.array(p_from_mdp_to_mrp )
r_from_mdp_to_mrp=[-0.5,-1.5,-1.0,5.5,0]

def compute(p, reward, gamma, states_num):
    """
    利用贝尔曼方程的矩阵形式
    :param p:
    :param reward:
    :param gamma:
    :param states_num:
    :return:
    """
    reward = np.array(reward).reshape((-1, 1))  # 将reward写成列向量的模型
    value = np.dot(np.linalg.inv(np.eye(states_num, states_num) - gamma * p), reward)
    return value

v = compute(p_from_mdp_to_mrp,r_from_mdp_to_mrp,gamma,5)
# 由状态价值函数计算动作价值函数

def sample(mdp,pi,timestep_max,number):
    S, A, P, R, gamma = MDP
    episode = []
    timestep = 0
    s = S[np.random.randint(4)]  # 随机选择一个除s5以外的状态s作为起点
    while s!="s5" and timestep<timestep_max:
        timestep += 0
        rand,temp = np.random.rand(),0

