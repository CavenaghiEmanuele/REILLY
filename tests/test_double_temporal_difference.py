import reilly as rl


def test_Double_QLearning():
    env = rl.Taxi()
    session = rl.PyBindSession(env)
    agent = rl.DoubleQLearning(
        states=env.states,
        actions=env.actions,
        alpha=0.1,
        epsilon=0.03,
        gamma=0.99
    )
    session.add_agent(agent)
    session.run(100, 10, 10)


def test_Double_Sarsa():
    env = rl.Taxi()
    session = rl.PyBindSession(env)
    agent = rl.DoubleSarsa(
        states=env.states,
        actions=env.actions,
        alpha=0.1,
        epsilon=0.03,
        gamma=0.99
    )
    session.add_agent(agent)
    session.run(100, 10, 10)


def test_Double_Expected_Sarsa():
    env = rl.Taxi()
    session = rl.PyBindSession(env)
    agent = rl.DoubleExpectedSarsa(
        states=env.states,
        actions=env.actions,
        alpha=0.1,
        epsilon=0.03,
        gamma=0.99
    )
    session.add_agent(agent)
    session.run(100, 10, 10)
