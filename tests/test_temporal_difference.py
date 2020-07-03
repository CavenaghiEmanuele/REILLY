import reilly as rl


def test_QLearning():
    env = rl.Frozen_Lake4x4()
    session = rl.PyBindSession(env)
    agent = rl.QLearning(
        states=env.states,
        actions=env.actions,
        alpha=0.1,
        epsilon=0.03,
        gamma=0.99
    )
    session.add_agent(agent)
    session.run(100, 10, 10)


def test_Sarsa():
    for E in rl.ENVIRONMENTS():
        env = E()
        env = rl.Frozen_Lake4x4()
        session = rl.PyBindSession(env)
        agent = rl.Sarsa(
            states=env.states,
            actions=env.actions,
            alpha=0.1,
            epsilon=0.03,
            gamma=0.99
        )
        session.add_agent(agent)
        session.run(100, 10, 10)


def test_Expected_Sarsa():
    env = rl.Frozen_Lake4x4()
    session = rl.PyBindSession(env)
    agent = rl.ExpectedSarsa(
        states=env.states,
        actions=env.actions,
        alpha=0.1,
        epsilon=0.03,
        gamma=0.99
    )
    session.add_agent(agent)
    session.run(100, 10, 10)
