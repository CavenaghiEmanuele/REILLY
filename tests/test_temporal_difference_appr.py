import reinforcement_learning as rl


def test_sarsa_appr_agent():
    env = rl.Frozen_Lake4x4()
    session = rl.Session(env)
    agent = rl.SarsaApproximateAgent(
        alpha=0.1,
        epsilon=0.03,
        gamma=0.99,
        feature_dims = 1,
        num_tilings=2,
        tiling_offset=[1],
        tiles_dims=[1]
    )
    session.add_agent(agent)
    session.reset_env()
    session.run(100, 10, 10)
    