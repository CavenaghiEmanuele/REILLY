import reinforcement_learning as rl


def test_Sarsa_agent():

    env = rl.Frozen_Lake4x4()
    agent = rl.QLearningAgent(alpha=0.1, epsilon=0.03, gamma=0.99, environment=env)

    agent.run(n_episodes=100, n_tests=10, test_step=10)
