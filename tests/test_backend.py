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
