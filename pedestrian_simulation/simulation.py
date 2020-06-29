from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pedestrian_simulation as ps
import reinforcement_learning as rl


def simulate(env, n_agents: int, start_step: int = 1):
    single_train_session = rl.SingleTrainSession(env, start_step=start_step)

    for _ in range(n_agents):
        agent = rl.SarsaLambdaAgent(
            alpha=0.1,
            epsilon=0.3,
            gamma=0.99,
            lambd=0.98,
            epsilon_decay=0.99,
            trace_type="replacing",
            feature_dims=2,
            num_tilings=4,
            tiles_size=[4, 4],
        )
        single_train_session.add_agent(agent)

    results = single_train_session.run(500, 500, 50)
    file_name = str(n_agents) + " agents - " + str(start_step) + " delay steps.csv"
    results.to_csv(file_name, index=False)
    #rl.plot(results)


def duplicated_simulation(env, n_train_agents, n_duplication: int, start_step: int = 1):

    single_train_session = rl.SingleTrainSession(env, start_step=start_step)
    trained_agents = []

    for _ in range(n_train_agents):
        agent = rl.SarsaLambdaAgent(
            alpha=0.1,
            epsilon=0.3,
            gamma=0.99,
            lambd=0.98,
            epsilon_decay=0.99,
            trace_type="replacing",
            feature_dims=2,
            num_tilings=4,
            tiles_size=[4, 4],
        )
        single_train_session.add_agent(agent)
        trained_agents.append(agent)
    
    single_train_session.run(500, 500, 1)

    for agent in trained_agents:
        for _ in range(n_duplication):
            copy_agent = rl.SarsaLambdaAgent(
                alpha=0.1,
                epsilon=0.3,
                gamma=0.99,
                lambd=0.98,
                epsilon_decay=0.99,
                trace_type="replacing",
                feature_dims=2,
                num_tilings=4,
                tiles_size=[4, 4],
            )
            copy_agent._Q_estimator = deepcopy(agent._Q_estimator)
            single_train_session.add_agent(copy_agent)

    results = single_train_session.run(10, 1, 50)
    rl.plot(results)

def travelling_time(results):
    groupby = results.drop(['return_sum', 'wins', 'time', 'distance'], axis=1).\
        groupby(['test', 'sample', 'agent'])
    arrival_time = groupby.max()
    travelling_time = arrival_time - groupby.min()
    scatter = pd.concat([arrival_time, travelling_time], axis=1, sort=False)
    points = [
        tuple(row)
        for _, row in scatter.loc[(1,49)].iterrows()
    ]
    plt.scatter([p[0] for p in points], [p[1] for p in points])
    plt.savefig(datetime.now().strftime("%Y%m%d_%H%M%S") + '_travelling_scatter.jpg')
    plt.clf()

def density(results, area, sample):
    groupby = results.drop(['return_sum', 'wins', 'time', 'distance'], axis=1).groupby(['test', 'sample', 'step'])
    count = groupby.count()
    density = count / area
    points = [
        tuple(row)
        for _, row in density.loc[(1,sample)].iterrows()
    ]
    plt.plot(list(range(len(points))), [p[0] for p in points])
    plt.savefig(datetime.now().strftime("%Y%m%d_%H%M%S") + '_density_chart.jpg')
    plt.clf()

def speed(results, sample):
    groupby = results.drop(['return_sum', 'wins', 'step'], axis=1).groupby(['test', 'sample', 'agent'])
    speed = groupby.sum()
    speed['speed'] = speed['distance'] / speed['time']
    speed = speed.drop(['distance', 'time'], axis=1).round(decimals=1)
    points = speed.loc[(1, sample)]['speed'].value_counts().to_dict()
    points = [
        (key, value)
        for key, value in points.items()
    ]
    plt.bar([p[0] for p in points], [p[1] for p in points], width=0.05, edgecolor="black")
    plt.savefig(datetime.now().strftime("%Y%m%d_%H%M%S") + '_speed_bars.jpg')
    plt.clf()

def time(results, sample):
    groupby = results.drop(['return_sum', 'wins', 'step', 'distance'], axis=1).groupby(['test', 'sample', 'agent'])
    time = groupby.sum()
    time = time.round(decimals=1)
    points = time.loc[(1, sample)]['time'].value_counts().to_dict()
    points = [
        (key, value)
        for key, value in points.items()
    ]
    plt.bar([p[0] for p in points], [p[1] for p in points], edgecolor="black")
    plt.savefig(datetime.now().strftime("%Y%m%d_%H%M%S") + '_time_bars.jpg')
    plt.clf()

def wall_clock_time(results, sample):
    groupby = results.drop(['return_sum', 'wins', 'time', 'distance', 'agent'], axis=1).groupby(['test', 'sample'])
    wall_clock_time = groupby.max() * 0.25
    wall_clock_time.columns = ['wall_clock_time']
    print(wall_clock_time.loc[(1, sample)])
    
def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(name)


if __name__ == "__main__":
    env = ps.japan()

    simulate(env, n_agents=46, start_step=20)
    '''
    results = load_csv("46 agents - 20 delay steps.csv")
    #sample = 1
    #density(results, 77, sample)
    #speed(results, sample)
    #time(results, sample)
    for sample in range(49):
        try:
            wall_clock_time(results, sample)
        except:
            pass
    '''