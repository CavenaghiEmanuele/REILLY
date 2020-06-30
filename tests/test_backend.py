import reinforcement_learning as rl


def test_MonteCarloFirstVisit():
    env = rl.Frozen_Lake4x4()
    session = rl.PyBindSession(env)
    agent = rl.backend.MonteCarloFirstVisit(
        states=env.states,
        actions=env.actions,
        epsilon=0.03,
        gamma=0.99
    )
    session.add_agent(agent)
    session.run(100, 10, 10)

def test_MonteCarloEveryVisit():
    env = rl.Frozen_Lake4x4()
    session = rl.PyBindSession(env)
    agent = rl.backend.MonteCarloEveryVisit(
        states=env.states,
        actions=env.actions,
        epsilon=0.03,
        gamma=0.99
    )
    session.add_agent(agent)
    session.run(100, 10, 10)

def test_Sarsa():
    env = rl.Frozen_Lake4x4()
    session = rl.PyBindSession(env)
    agent = rl.backend.Sarsa(
        states=env.states,
        actions=env.actions,
        alpha=0.1,
        epsilon=0.03,
        gamma=0.99
    )
    session.add_agent(agent)
    session.run(100, 10, 10)

def test_QLearning():
    env = rl.Frozen_Lake4x4()
    session = rl.PyBindSession(env)
    agent = rl.backend.QLearning(
        states=env.states,
        actions=env.actions,
        alpha=0.1,
        epsilon=0.03,
        gamma=0.99
    )
    session.add_agent(agent)
    session.run(100, 10, 10)

def test_ExpectedSarsa():
    env = rl.Frozen_Lake4x4()
    session = rl.PyBindSession(env)
    agent = rl.backend.ExpectedSarsa(
        states=env.states,
        actions=env.actions,
        alpha=0.1,
        epsilon=0.03,
        gamma=0.99
    )
    session.add_agent(agent)
    session.run(100, 10, 10)

def test_DoubleSarsa():
    env = rl.Frozen_Lake4x4()
    session = rl.PyBindSession(env)
    agent = rl.backend.DoubleSarsa(
        states=env.states,
        actions=env.actions,
        alpha=0.1,
        epsilon=0.03,
        gamma=0.99
    )
    session.add_agent(agent)
    session.run(100, 10, 10)

def test_DoubleQLearning():
    env = rl.Frozen_Lake4x4()
    session = rl.PyBindSession(env)
    agent = rl.backend.DoubleQLearning(
        states=env.states,
        actions=env.actions,
        alpha=0.1,
        epsilon=0.03,
        gamma=0.99
    )
    session.add_agent(agent)
    session.run(100, 10, 10)

def test_DoubleExpectedSarsa():
    env = rl.Frozen_Lake4x4()
    session = rl.PyBindSession(env)
    agent = rl.backend.DoubleExpectedSarsa(
        states=env.states,
        actions=env.actions,
        alpha=0.1,
        epsilon=0.03,
        gamma=0.99
    )
    session.add_agent(agent)
    session.run(100, 10, 10)

def test_NStepSarsa():
    env = rl.Frozen_Lake4x4()
    session = rl.PyBindSession(env)
    agent = rl.backend.NStepSarsa(
        states=env.states,
        actions=env.actions,
        alpha=0.1,
        epsilon=0.03,
        gamma=0.99,
        n_step=5
    )
    session.add_agent(agent)
    session.run(100, 10, 10)

def test_NStepExpectedSarsa():
    env = rl.Frozen_Lake4x4()
    session = rl.PyBindSession(env)
    agent = rl.backend.NStepExpectedSarsa(
        states=env.states,
        actions=env.actions,
        alpha=0.1,
        epsilon=0.03,
        gamma=0.99,
        n_step=5
    )
    session.add_agent(agent)
    session.run(100, 10, 10)

def test_NStepTreeBackup():
    env = rl.Frozen_Lake4x4()
    session = rl.PyBindSession(env)
    agent = rl.backend.NStepTreeBackup(
        states=env.states,
        actions=env.actions,
        alpha=0.1,
        epsilon=0.03,
        gamma=0.99,
        n_step=5
    )
    session.add_agent(agent)
    session.run(100, 10, 10)

def test_TabularDynaQ():
    env = rl.Frozen_Lake4x4()
    session = rl.PyBindSession(env)
    agent = rl.backend.TabularDynaQ(
        states=env.states,
        actions=env.actions,
        alpha=0.1,
        epsilon=0.03,
        gamma=0.99,
        n_plan=5
    )
    session.add_agent(agent)
    session.run(100, 10, 10)

def test_TabularDynaQPlus():
    env = rl.Frozen_Lake4x4()
    session = rl.PyBindSession(env)
    agent = rl.backend.TabularDynaQPlus(
        states=env.states,
        actions=env.actions,
        alpha=0.1,
        epsilon=0.03,
        gamma=0.99,
        n_plan=5
    )
    session.add_agent(agent)
    session.run(100, 10, 10)

def test_PrioritizedSweeping():
    env = rl.Frozen_Lake4x4()
    session = rl.PyBindSession(env)
    agent = rl.backend.PrioritizedSweeping(
        states=env.states,
        actions=env.actions,
        alpha=0.1,
        epsilon=0.03,
        gamma=0.99,
        n_plan=5,
        theta=0.5
    )
    session.add_agent(agent)
    session.run(100, 10, 10)

def test_SemiGradientMonteCarlo():
    env = rl.Frozen_Lake4x4()
    session = rl.PyBindSession(env)
    agent = rl.backend.SemiGradientMonteCarlo(
        actions=env.actions,
        alpha=0.1,
        epsilon=0.01,
        gamma=0.99,
        features=1,
        tilings=2,
        tilings_offset=[4, 4],
        tile_size=[4, 4]
    )
    session.add_agent(agent)
    session.run(100, 10, 10)

def test_SemiGradientSarsa():
    env = rl.Frozen_Lake4x4()
    session = rl.PyBindSession(env)
    agent = rl.backend.SemiGradientExpectedSarsa(
        actions=env.actions,
        alpha=0.1,
        epsilon=0.01,
        gamma=0.99,
        features=1,
        tilings=2,
        tilings_offset=[4, 4],
        tile_size=[4, 4]
    )
    session.add_agent(agent)
    session.run(100, 10, 10)

def test_SemiGradientExpectedSarsa():
    env = rl.Frozen_Lake4x4()
    session = rl.PyBindSession(env)
    agent = rl.backend.SemiGradientExpectedSarsa(
        actions=env.actions,
        alpha=0.1,
        epsilon=0.01,
        gamma=0.99,
        features=1,
        tilings=2,
        tilings_offset=[4, 4],
        tile_size=[4, 4]
    )
    session.add_agent(agent)
    session.run(100, 10, 10)