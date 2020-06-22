import reinforcement_learning as rl


def test_sarsa_lambda():
    env = rl.Frozen_Lake4x4()
    session = rl.JointTrainSession(env)
    agent = rl.SarsaLambdaAgent(
            alpha=0.1,
            epsilon=0.03,
            gamma=0.99,
            lambd=0.98,
            have_trace=True,
            trace_type="replacing",
            feature_dims=1,
            num_tilings=4,
        )
    session.add_agent(agent)
    session.run(100, 10, 10)
