import matplotlib.pyplot as plt

import reinforcement_learning as rl


def plot_results(results: dict):
    for test in results.keys():
        plt.figure(test)
        plt.plot(results[test])
        plt.ylabel(test)
        plt.xlabel("Number of tests")
        plt.grid(linestyle="--", linewidth=0.5, color='.25', zorder=-10)
    plt.show()

if __name__ == "__main__":
    
    env = rl.Taxi()
    agent = rl.DoubleQLearningAgent(alpha=0.1, epsilon=0.05, gamma=0.99, environment=env)

    print(agent)
    #agent.train(n_episodes=10000)
    #agent.test(10000)
    
    results = agent.run(n_episodes=10000, n_tests=50, test_step=100)
    plot_results(results)
