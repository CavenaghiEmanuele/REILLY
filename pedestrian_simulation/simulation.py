import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import reinforcement_learning as rl
import pedestrian_simulation as ps


def simulate(env, n_agents: int, start_step:int = 1):
    
    single_train_session = rl.SingleTrainSession(env, start_step=start_step)
    
    for _ in range(n_agents):
        agent = rl.SarsaLambdaAgent(
                alpha=0.1,
                epsilon=0.3,
                gamma=0.99,
                lambd=0.98,
                epsilon_decay=0.99,
                trace_type="replacing",
                feature_dims=2,
                num_tilings=4,
                tiles_size=[4,4],
            )
        single_train_session.add_agent(agent)

    results = single_train_session.run(500, 50, 50)
    rl.plot(results)
    

if __name__ == "__main__":
    
    env = ps.T()
    simulate(env, n_agents=20, start_step=4)
