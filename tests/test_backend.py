import reilly as rl


def test_MonteCarloFirstVisit():
    env = rl.Taxi()
    agent = rl.backend.MonteCarloFirstVisit(
        states=env.states,
        actions=env.actions,
        epsilon=0.03,
        gamma=0.99
    )
    session = rl.Session(env, agent)
    session.run(100, 10, 10)

def test_MonteCarloEveryVisit():
    env = rl.Taxi()
    agent = rl.backend.MonteCarloEveryVisit(
        states=env.states,
        actions=env.actions,
        epsilon=0.03,
        gamma=0.99
    )
    session = rl.Session(env, agent)
    session.run(100, 10, 10)

def test_Sarsa():
    env = rl.Taxi()
    agent = rl.Sarsa(
        states=env.states,
        actions=env.actions,
        alpha=0.1,
        epsilon=0.03,
        gamma=0.99,
        backend='cpp'
    )
    session = rl.Session(env, agent)
    session.run(100, 10, 10)

def test_QLearning():
    env = rl.Taxi()
    agent = rl.QLearning(
        states=env.states,
        actions=env.actions,
        alpha=0.1,
        epsilon=0.03,
        gamma=0.99,
        backend='cpp'
    )
    session = rl.Session(env, agent)
    session.run(100, 10, 10)

def test_ExpectedSarsa():
    env = rl.Taxi()
    agent = rl.ExpectedSarsa(
        states=env.states,
        actions=env.actions,
        alpha=0.1,
        epsilon=0.03,
        gamma=0.99,
        backend='cpp'
    )
    session = rl.Session(env, agent)
    session.run(100, 10, 10)

def test_DoubleSarsa():
    env = rl.Taxi()
    agent = rl.DoubleSarsa(
        states=env.states,
        actions=env.actions,
        alpha=0.1,
        epsilon=0.03,
        gamma=0.99,
        backend='cpp'
    )
    session = rl.Session(env, agent)
    session.run(100, 10, 10)

def test_DoubleQLearning():
    env = rl.Taxi()
    agent = rl.DoubleQLearning(
        states=env.states,
        actions=env.actions,
        alpha=0.1,
        epsilon=0.03,
        gamma=0.99,
        backend='cpp'
    )
    session = rl.Session(env, agent)
    session.run(100, 10, 10)

def test_DoubleExpectedSarsa():
    env = rl.Taxi()
    agent = rl.DoubleExpectedSarsa(
        states=env.states,
        actions=env.actions,
        alpha=0.1,
        epsilon=0.03,
        gamma=0.99,
        backend='cpp'
    )
    session = rl.Session(env, agent)
    session.run(100, 10, 10)

def test_NStepSarsa():
    env = rl.Taxi()
    agent = rl.NStepSarsa(
        states=env.states,
        actions=env.actions,
        alpha=0.1,
        epsilon=0.03,
        gamma=0.99,
        n_step=5,
        backend='cpp'
    )
    session = rl.Session(env, agent)
    session.run(100, 10, 10)

def test_NStepExpectedSarsa():
    env = rl.Taxi()
    agent = rl.NStepExpectedSarsa(
        states=env.states,
        actions=env.actions,
        alpha=0.1,
        epsilon=0.03,
        gamma=0.99,
        n_step=5,
        backend='cpp'
    )
    session = rl.Session(env, agent)
    session.run(100, 10, 10)

def test_NStepTreeBackup():
    env = rl.Taxi()
    agent = rl.NStepTreeBackup(
        states=env.states,
        actions=env.actions,
        alpha=0.1,
        epsilon=0.03,
        gamma=0.99,
        n_step=5,
        backend='cpp'
    )
    session = rl.Session(env, agent)
    session.run(100, 10, 10)

def test_TabularDynaQ():
    env = rl.Taxi()
    agent = rl.TabularDynaQ(
        states=env.states,
        actions=env.actions,
        alpha=0.1,
        epsilon=0.03,
        gamma=0.99,
        n_plan=5,
        backend='cpp'
    )
    session = rl.Session(env, agent)
    session.run(100, 10, 10)

def test_TabularDynaQPlus():
    env = rl.Taxi()
    agent = rl.TabularDynaQPlus(
        states=env.states,
        actions=env.actions,
        alpha=0.1,
        epsilon=0.03,
        gamma=0.99,
        n_plan=5,
        backend='cpp'
    )
    session = rl.Session(env, agent)
    session.run(100, 10, 10)

def test_PrioritizedSweeping():
    env = rl.Taxi()
    agent = rl.PrioritizedSweeping(
        states=env.states,
        actions=env.actions,
        alpha=0.1,
        epsilon=0.03,
        gamma=0.99,
        n_plan=5,
        theta=0.5,
        backend='cpp'
    )
    session = rl.Session(env, agent)
    session.run(100, 10, 10)

def test_SemiGradientMonteCarlo():
    env = rl.Taxi()
    agent = rl.SemiGradientMonteCarlo(
        actions=env.actions,
        alpha=0.1,
        epsilon=0.01,
        gamma=0.99,
        features=1,
        tilings=2,
        tilings_offset=[4, 4],
        tile_size=[4, 4],
        backend='cpp'
    )
    session = rl.Session(env, agent)
    session.run(100, 10, 10)

def test_SemiGradientSarsa():
    env = rl.Taxi()
    agent = rl.SemiGradientSarsa(
        actions=env.actions,
        alpha=0.1,
        epsilon=0.01,
        gamma=0.99,
        features=1,
        tilings=2,
        tilings_offset=[4, 4],
        tile_size=[4, 4],
        backend='cpp'
    )
    session = rl.Session(env, agent)
    session.run(100, 10, 10)

def test_SemiGradientExpectedSarsa():
    env = rl.Taxi()
    agent = rl.SemiGradientExpectedSarsa(
        actions=env.actions,
        alpha=0.1,
        epsilon=0.01,
        gamma=0.99,
        features=1,
        tilings=2,
        tilings_offset=[4, 4],
        tile_size=[4, 4],
        backend='cpp'
    )
    session = rl.Session(env, agent)
    session.run(100, 10, 10)

def test_SemiGradientNStepSarsa():
    env = rl.Taxi()
    agent = rl.SemiGradientNStepSarsa(
        actions=env.actions,
        alpha=0.1,
        epsilon=0.01,
        gamma=0.99,
        n_step=5,
        features=1,
        tilings=2,
        tilings_offset=[4, 4],
        tile_size=[4, 4],
        backend='cpp'
    )
    session = rl.Session(env, agent)
    session.run(100, 10, 10)

def test_SemiGradientNStepExpectedSarsa():
    env = rl.Taxi()
    agent = rl.SemiGradientNStepExpectedSarsa(
        actions=env.actions,
        alpha=0.1,
        epsilon=0.01,
        gamma=0.99,
        n_step=5,
        features=1,
        tilings=2,
        tilings_offset=[4, 4],
        tile_size=[4, 4],
        backend='cpp'
    )
    session = rl.Session(env, agent)
    session.run(100, 10, 10)
