import numpy as np
np.random.seed(0)
# 定义状态转移矩阵
p = [[0.9,0.1,0.0,0.0,0.0,0.0], # 节点1到它自己是0.9，到2是0.1
     [0.5,0.0,0.5,0.0,0.0,0.0], # 节点2 到1和到3是0.5
     [0.0,0.0,0.0,0.6,0.0,0.4],
     [0.0,0.0,0.0,0.0,0.3,0.7],
     [0.0,0.2,0.3,0.5,0.0,0.0],
     [0.0,0.0,0.0,0.0,0.0,1.0],]
p = np.array(p)
# ------------------------------------------------------------计算回报---------------------------------------------------
# 定义奖励函数
reward = [-1, -2, -2, 10, 1, 0]
# 定义折扣
gamma = 0.5
def compute_return(start_index,chain,gamma):
    G = 0
    for i in reversed(range(start_index,len(chain))): # 为了凑这个衰减
         # print(reward[chain[i]-1])
         G = gamma*G+reward[chain[i]-1]
    return G

# 一个序列s1-s2-s3-s6
chain = [1,2,3,6]
start_index = 0
G = compute_return(start_index,chain,gamma)
print(G) # 这条链的gain
# 价值函数
# ---------------------------------------价值函数------------------------------------------------------------------------
# 一个状态的期望回报就是价值函数

def compute(p,reward,gamma,states_num):
     """
     利用贝尔曼方程的矩阵形式
     :param p:
     :param reward:
     :param gamma:
     :param states_num:
     :return:
     """
     reward = np.array(reward).reshape((-1,1))# 将reward写成列向量的模型
     value = np.dot(np.linalg.inv(np.eye(states_num,states_num)-gamma*p),reward)
     return value
states_num = 6
v = compute(p,reward,gamma,states_num)
print(v)