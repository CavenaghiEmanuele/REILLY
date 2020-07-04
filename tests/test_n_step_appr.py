import reilly as rl


def test_semigradient_n_step_sarsa():
    env = rl.Taxi()
    agent = rl.SemiGradientNStepSarsa(
        actions=env.actions,
        alpha=0.1,
        epsilon=0.03,
        gamma=0.99,
        n_step=5,
        features=1,
        tilings=2,
        tilings_offset=[1],
        tile_size=[1]
    )
    session = rl.Session(env, agent)
    session.run(100, 10, 10)
    
def test_semigradient_n_step_expected_sarsa():
    env = rl.Taxi()
    agent = rl.SemiGradientNStepExpectedSarsa(
        actions=env.actions,
        alpha=0.1,
        epsilon=0.03,
        gamma=0.99,
        n_step=5,
        features=1,
        tilings=2,
        tilings_offset=[1],
        tile_size=[1]
    )
    session = rl.Session(env, agent)
    session.run(100, 10, 10)
    