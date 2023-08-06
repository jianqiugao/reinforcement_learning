import matplotlib.pyplot as plt
def plot_results(solvers, solver_name):
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_name[idx])
    plt.xlabel('time_steps')
    plt.ylabel('cumulative regret ')
    plt.legend()
    plt.show()