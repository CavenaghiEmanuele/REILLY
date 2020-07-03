import reilly as rl


def test_MonteCarlo():
    env = rl.Taxi()
    session = rl.PyBindSession(env)
    agent0 = rl.MonteCarlo(
        states=env.states,
        actions=env.actions,
        epsilon=0.03,
        gamma=0.99,
        visit_update='first'
    )
    agent1 = rl.MonteCarlo(
        states=env.states,
        actions=env.actions,
        epsilon=0.03,
        gamma=0.99,
        visit_update='every'
    )
    session.add_agent(agent0)
    session.add_agent(agent1)
    session.run(100, 10, 10)
