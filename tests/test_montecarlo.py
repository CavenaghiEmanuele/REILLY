import reilly as rl


def test_MonteCarlo():
    env = rl.Taxi()

    agent = rl.MonteCarlo(
        states=env.states,
        actions=env.actions,
        epsilon=0.03,
        gamma=0.99,
        visit_update='first'
    )
    session = rl.Session(env, agent)
    session.run(100, 10, 10)

    agent = rl.MonteCarlo(
        states=env.states,
        actions=env.actions,
        epsilon=0.03,
        gamma=0.99,
        visit_update='every'
    )
    session = rl.Session(env, agent)
    session.run(100, 10, 10)
