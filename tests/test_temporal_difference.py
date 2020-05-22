import reinforcement_learning as rl


def test_QLearning_agent():
    env = rl.Frozen_Lake4x4()
    session = rl.Session(env)
    agent = rl.QLearningAgent(
        states_size=env.states_size(),
        actions_size=env.actions_size(),
        alpha=0.1,
        epsilon=0.03,
        gamma=0.99
    )
    session.add_agent(agent)
    session.reset_env()
    session.run(100, 10, 10)


def test_Sarsa_agent():
    env = rl.Frozen_Lake4x4()
    session = rl.Session(env)
    agent = rl.SarsaAgent(
        states_size=env.states_size(),
        actions_size=env.actions_size(),
        alpha=0.1,
        epsilon=0.03,
        gamma=0.99
    )
    session.add_agent(agent)
    session.reset_env()
    session.run(100, 10, 10)


def test_Expected_Sarsa_agent():
    env = rl.Frozen_Lake4x4()
    session = rl.Session(env)
    agent = rl.ExpectedSarsaAgent(
        states_size=env.states_size(),
        actions_size=env.actions_size(),
        alpha=0.1,
        epsilon=0.03,
        gamma=0.99
    )
    session.add_agent(agent)
    session.reset_env()
    session.run(100, 10, 10)
