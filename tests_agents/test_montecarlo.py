import reinforcement_learning as rl


def test_MonteCarlo_agent():

    env = rl.Frozen_Lake4x4()
    agent = rl.MonteCarloAgent(epsilon=0.03, gamma=0.99, environment=env, visit_update="first")

    agent.run(n_episodes=100, n_tests=10, test_step=10)