from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt; plt.style.use('seaborn-poster')
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

    results = single_train_session.run(500, 50, 50)
    results.to_csv("Test.csv", index=False)
    rl.plot(results)


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
        for _, row in scatter.iterrows()
    ]
    plt.scatter([p[0] for p in points], [p[1] for p in points])
    plt.savefig(datetime.now().strftime("%Y%m%d_%H%M%S") + '_travelling_scatter.jpg')

def density(results, area):
    groupby = results.drop(['return_sum', 'wins', 'time', 'distance'], axis=1).groupby(['test', 'sample', 'step'])
    count = groupby.count()
    density = count / area
    points = [
        tuple(row)
        for _, row in density.iterrows()
    ]
    plt.plot(list(range(len(points))), [p[0] for p in points])
    plt.savefig(datetime.now().strftime("%Y%m%d_%H%M%S") + '_density_chart.jpg')

def speed(results):
    groupby = results.drop(['return_sum', 'wins', 'step'], axis=1).groupby(['test', 'sample', 'agent'])
    speed = groupby.sum()
    speed['speed'] = speed['distance'] / speed['time']
    speed = speed.drop(['distance', 'time'], axis=1).round(decimals=1)
    points = speed.loc[(10)]['speed'].value_counts().to_dict()
    points = [
        (key, value)
        for key, value in points.items()
    ]
    plt.bar([p[0] for p in points], [p[1] for p in points], width=0.05, edgecolor="black")
    plt.savefig(datetime.now().strftime("%Y%m%d_%H%M%S") + '_speed_bars.jpg')
    
def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(name)



if __name__ == "__main__":
    env = ps.three_room()
    #simulate(env, n_agents=1, start_step=1)
    #duplicated_simulation(env, n_train_agents=20, n_duplication=100, start_step=250)
    
    results = load_csv("Test.csv")
    rl.plot(results)
    
    