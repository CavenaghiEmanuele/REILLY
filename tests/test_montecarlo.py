import reinforcement_learning as rl


def test_MonteCarlo_agent():
    env = rl.Frozen_Lake4x4()
    session = rl.JointTrainSession(env)
    agent0 = rl.MonteCarloAgent(
        states=env.states,
        actions=env.actions,
        epsilon=0.03,
        gamma=0.99,
        visit_update='first'
    )
    agent1 = rl.MonteCarloAgent(
        states=env.states,
        actions=env.actions,
        epsilon=0.03,
        gamma=0.99,
        visit_update='every'
    )
    session.add_agent(agent0)
    session.add_agent(agent1)
    session.run(100, 10, 10)
