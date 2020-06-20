import reinforcement_learning as rl


def test_Double_QLearning_agent():
    env = rl.Frozen_Lake4x4()
    session = rl.JointTrainSession(env)
    agent = rl.DoubleQLearningAgent(
        states_size=env.states_size,
        actions_size=env.actions_size,
        alpha=0.1,
        epsilon=0.03,
        gamma=0.99
    )
    session.add_agent(agent)
    session.run(100, 10, 10)


def test_Double_Sarsa_agent():
    env = rl.Frozen_Lake4x4()
    session = rl.JointTrainSession(env)
    agent = rl.DoubleSarsaAgent(
        states_size=env.states_size,
        actions_size=env.actions_size,
        alpha=0.1,
        epsilon=0.03,
        gamma=0.99
    )
    session.add_agent(agent)
    session.run(100, 10, 10)


def test_Double_Expected_Sarsa_agent():
    env = rl.Frozen_Lake4x4()
    session = rl.JointTrainSession(env)
    agent = rl.DoubleExpectedSarsaAgent(
        states_size=env.states_size,
        actions_size=env.actions_size,
        alpha=0.1,
        epsilon=0.03,
        gamma=0.99
    )
    session.add_agent(agent)
    session.run(100, 10, 10)
