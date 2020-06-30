import reilly as rl


def test_n_step_sarsa_agent():
    env = rl.Frozen_Lake4x4()
    session = rl.JointTrainSession(env)
    agent = rl.NStepSarsaAgent(
        states=env.states,
        actions=env.actions,
        alpha=0.1,
        epsilon=0.03,
        gamma=0.99,
        n_step=5
    )
    session.add_agent(agent)
    session.run(100, 10, 10)


def test_n_step_expected_sarsa_agent():
    env = rl.Frozen_Lake4x4()
    session = rl.JointTrainSession(env)
    agent = rl.NStepExpectedSarsaAgent(
        states=env.states,
        actions=env.actions,
        alpha=0.1,
        epsilon=0.03,
        gamma=0.99,
        n_step=5
    )
    session.add_agent(agent)
    session.run(100, 10, 10)
