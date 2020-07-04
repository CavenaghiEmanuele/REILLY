import reilly as rl


def test_n_step_sarsa():
    env = rl.Taxi()
    agent = rl.NStepSarsa(
        states=env.states,
        actions=env.actions,
        alpha=0.1,
        epsilon=0.03,
        gamma=0.99,
        n_step=5
    )
    session = rl.Session(env, agent)
    session.run(100, 10, 10)


def test_n_step_expected_sarsa():
    env = rl.Taxi()
    agent = rl.NStepExpectedSarsa(
        states=env.states,
        actions=env.actions,
        alpha=0.1,
        epsilon=0.03,
        gamma=0.99,
        n_step=5
    )
    session = rl.Session(env, agent)
    session.run(100, 10, 10)
