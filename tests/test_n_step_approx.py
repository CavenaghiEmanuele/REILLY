import reinforcement_learning as rl


def test_n_step_sarsa_appr_agent():
    env = rl.Frozen_Lake4x4()
    session = rl.JointTrainSession(env)
    agent1 = rl.NStepSarsaApproximateAgent(
        alpha=0.1,
        epsilon=0.03,
        gamma=0.99,
        n_step=5,
        feature_dims=1,
        num_tilings=2,
        tiling_offset=[1],
        tiles_size=[1]
    )
    
    agent2 = rl.NStepSarsaApproximateAgent(
        alpha=0.1,
        epsilon=0.03,
        gamma=0.99,
        n_step=5,
        feature_dims=1,
        num_tilings=2,
        tiling_offset=1,
        tiles_size=1
    )
    
    agent3 = rl.NStepSarsaApproximateAgent(
        alpha=0.1,
        epsilon=0.03,
        gamma=0.99,
        n_step=5,
        feature_dims=1,
        num_tilings=2
    )
    session.add_agent(agent1)
    session.add_agent(agent2)
    session.add_agent(agent3)
    session.run(100, 10, 10)
    
def test_n_step_expected_sarsa_appr_agent():
    env = rl.Frozen_Lake4x4()
    session = rl.JointTrainSession(env)
    agent = rl.NStepExpectedSarsaApproximateAgent(
        alpha=0.1,
        epsilon=0.03,
        gamma=0.99,
        n_step=5,
        feature_dims=1,
        num_tilings=2,
        tiling_offset=[1],
        tiles_size=[1]
    )
    session.add_agent(agent)
    session.run(100, 10, 10)
    