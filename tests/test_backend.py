import reinforcement_learning as rl


def test_MonteCarloFirstVisit():
    env = rl.Frozen_Lake4x4()
    session = rl.PyBindSession(env)
    agent = rl.backend.MonteCarloFirstVisit(
        states=env.states,
        actions=env.actions,
        epsilon=0.03,
        gamma=0.99
    )
    session.add_agent(agent)
    session.run(100, 10, 10)

def test_MonteCarloEveryVisit():
    env = rl.Frozen_Lake4x4()
    session = rl.PyBindSession(env)
    agent = rl.backend.MonteCarloEveryVisit(
        states=env.states,
        actions=env.actions,
        epsilon=0.03,
        gamma=0.99
    )
    session.add_agent(agent)
    session.run(100, 10, 10)

def test_Sarsa():
    env = rl.Frozen_Lake4x4()
    session = rl.PyBindSession(env)
    agent = rl.backend.Sarsa(
        states=env.states,
        actions=env.actions,
        alpha=0.1,
        epsilon=0.03,
        gamma=0.99
    )
    session.add_agent(agent)
    session.run(100, 10, 10)

def test_QLearning():
    env = rl.Frozen_Lake4x4()
    session = rl.PyBindSession(env)
    agent = rl.backend.QLearning(
        states=env.states,
        actions=env.actions,
        alpha=0.1,
        epsilon=0.03,
        gamma=0.99
    )
    session.add_agent(agent)
    session.run(100, 10, 10)

def test_ExpectedSarsa():
    env = rl.Frozen_Lake4x4()
    session = rl.PyBindSession(env)
    agent = rl.backend.ExpectedSarsa(
        states=env.states,
        actions=env.actions,
        alpha=0.1,
        epsilon=0.03,
        gamma=0.99
    )
    session.add_agent(agent)
    session.run(100, 10, 10)

def test_DoubleSarsa():
    env = rl.Frozen_Lake4x4()
    session = rl.PyBindSession(env)
    agent = rl.backend.DoubleSarsa(
        states=env.states,
        actions=env.actions,
        alpha=0.1,
        epsilon=0.03,
        gamma=0.99
    )
    session.add_agent(agent)
    session.run(100, 10, 10)

def test_DoubleQLearning():
    env = rl.Frozen_Lake4x4()
    session = rl.PyBindSession(env)
    agent = rl.backend.DoubleQLearning(
        states=env.states,
        actions=env.actions,
        alpha=0.1,
        epsilon=0.03,
        gamma=0.99
    )
    session.add_agent(agent)
    session.run(100, 10, 10)

def test_DoubleExpectedSarsa():
    env = rl.Frozen_Lake4x4()
    session = rl.PyBindSession(env)
    agent = rl.backend.DoubleExpectedSarsa(
        states=env.states,
        actions=env.actions,
        alpha=0.1,
        epsilon=0.03,
        gamma=0.99
    )
    session.add_agent(agent)
    session.run(100, 10, 10)
