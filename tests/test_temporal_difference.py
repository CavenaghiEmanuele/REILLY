import reilly as rl


def test_QLearning():
    env = rl.Taxi()
    agent = rl.QLearning(
        states=env.states,
        actions=env.actions,
        alpha=0.1,
        epsilon=0.03,
        gamma=0.99
    )
    session = rl.Session(env, agent)
    session.run(100, 10, 10)


def test_Sarsa():
    env = rl.Taxi()
    agent = rl.Sarsa(
        states=env.states,
        actions=env.actions,
        alpha=0.1,
        epsilon=0.03,
        gamma=0.99
    )
    session = rl.Session(env, agent)
    session.run(100, 10, 10)


def test_Expected_Sarsa():
    env = rl.Taxi()
    agent = rl.ExpectedSarsa(
        states=env.states,
        actions=env.actions,
        alpha=0.1,
        epsilon=0.03,
        gamma=0.99
    )
    session = rl.Session(env, agent)
    session.run(100, 10, 10)
