import reinforcement_learning as rl

if __name__ == "__main__":
    env = rl.Taxi()
    sess = rl.Session(env)
    agent = rl.DoubleQLearningAgent(
        states_size=env.states_size(),
        actions_size=env.actions_size(),
        alpha=0.1,
        epsilon=0.05,
        gamma=0.99
    )
    sess.add_agent(agent)
    sess.reset_env()
    results = sess.run(10000, 100, 50)
    rl.utils.plot(results)
