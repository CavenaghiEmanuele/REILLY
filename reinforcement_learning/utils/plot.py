import matplotlib.pyplot as plt

def plot(results):
    results = results.drop('step', axis=1).\
        groupby(['test', 'sample', 'agent']).sum()
    results = results.groupby(['test', 'agent']).mean()
    tests = results.columns.values.tolist()
    for test in tests:
        agents = results.groupby('agent')[test].apply(list).to_dict()
        plt.figure(test)
        for agent in agents:
            plt.plot(agents[agent])
        plt.ylabel(test)
        plt.xlabel("Number of tests")
        plt.grid(linestyle="--", linewidth=0.5, color='.25', zorder=-10)

    plt.show()
