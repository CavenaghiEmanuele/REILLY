import reinforcement_learning as rl


def test_n_step_sarsa_agent():
    env = rl.Frozen_Lake4x4()
    session = rl.Session(env)
    agent = rl.NStepSarsaAgent(
        states_size=env.states_size(),
        actions_size=env.actions_size(),
        alpha=0.1,
        epsilon=0.03,
        gamma=0.99,
        n_step=5
    )
    session.add_agent(agent)
    session.reset_env()
    session.run(100, 10, 10)
