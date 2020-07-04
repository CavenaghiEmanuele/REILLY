import reilly as rl


def test_sarsa_lambda():
    env = rl.Taxi()
    agent = rl.SarsaLambda(
            actions=env.actions,
            alpha=0.1,
            epsilon=0.03,
            gamma=0.99,
            lambd=0.98,
            trace_type="replacing",
            features=1,
            tilings=4,
        )
    session = rl.Session(env, agent)
    session.run(100, 10, 10)
