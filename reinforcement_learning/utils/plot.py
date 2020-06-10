import pandas as pd
import matplotlib.pyplot as plt

def plot(data: pd.DataFrame): 
    data = data.drop('step', axis=1).\
        groupby(['test', 'sample', 'agent']).sum()
    data = data.groupby(['test', 'agent']).mean()
    tests = data.columns.values.tolist()
    data = data.groupby('agent')
    for test in tests:
        agents = data[test].apply(list).to_dict()
        plt.figure(test)
        for agent in agents:
            plt.plot(agents[agent], label=agent)
        plt.ylabel(test)
        plt.xlabel('Number of tests')
        plt.grid(linestyle='--', linewidth=0.5, color='.25', zorder=-10)
    plt.legend(loc='upper left')
    plt.show()
