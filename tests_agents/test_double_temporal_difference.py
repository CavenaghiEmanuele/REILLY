import reinforcement_learning as rl


def test_Double_QLearning_agent():
    env = rl.Frozen_Lake4x4()
    agent = rl.DoubleQLearningAgent(alpha=0.1, epsilon=0.03, gamma=0.99, environment=env)
    agent.run(n_episodes=100, n_tests=10, test_step=10)

def test_Double_Sarsa_agent():
    env = rl.Frozen_Lake4x4()
    agent = rl.DoubleSarsaAgent(alpha=0.1, epsilon=0.03, gamma=0.99, environment=env)
    agent.run(n_episodes=100, n_tests=10, test_step=10)

def test_Double_Expected_Sarsa_agent():
    env = rl.Frozen_Lake4x4()
    agent = rl.DoubleExpectedSarsaAgent(alpha=0.1, epsilon=0.03, gamma=0.99, environment=env)
    agent.run(n_episodes=100, n_tests=10, test_step=10)
    