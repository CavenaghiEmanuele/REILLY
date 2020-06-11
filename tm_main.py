#!/usr/bin/env python3
import reinforcement_learning as rl

import pandas as pd
from tqdm import tqdm


if __name__ == "__main__":
       
    '''
    env = rl.TextEnvironment(
        text="####################\n" +
             "SS                  \n" +
             "SS                  \n" + 
             "                  X \n" +          
             "                    \n" +
             "####################",
        max_steps=1000,
        neighbor=rl.TextNeighbor.NEUMANN)    
    agent = rl.SarsaAgent(
            states_size=env.states_size,
            actions_size=env.actions_size,
            alpha=0.1,
            epsilon=0.2,
            gamma=0.99
        )
    agent2 = rl.QLearningAgent(
            states_size=env.states_size,
            actions_size=env.actions_size,
            alpha=0.1,
            epsilon=0.03,
            gamma=0.99
        )
    agent3 = rl.ExpectedSarsaAgent(
            states_size=env.states_size,
            actions_size=env.actions_size,
            alpha=0.1,
            epsilon=0.03,
            gamma=0.99
        )
    #Train first agent
    train_session = rl.Session(env)
    train_session.add_agent(agent)
    train_session.reset_env()
    train_result = train_session.run(1000, 100, 50)
    #rl.plot(train_result)
    
    #Train second agent
    train_session = rl.Session(env)
    train_session.add_agent(agent2)
    train_session.reset_env()
    train_result = train_session.run(1000, 100, 50)
    #rl.plot(train_result)
    
    #Train third agent
    train_session = rl.Session(env)
    train_session.add_agent(agent3)
    train_session.reset_env()
    train_result = train_session.run(1000, 100, 50)
    #rl.plot(train_result)
    
    
    test_session = rl.Session(env)
    test_session.add_agent(agent)
    test_session.add_agent(agent2)
    test_session.add_agent(agent3)        
    test_session.reset_env()
    
    out = []
    for episode in tqdm(range(100)):
        out.append(
            test_session._run_test(episode, 50, False)
        )
    
    rl.plot(pd.concat(out))
    '''
    
    
    env = rl.TextEnvironment(
        text="####################\n" +
             "SS                  \n" +
             "SS                  \n" + 
             "                  X \n" +          
             "                    \n" +
             "####################",
        max_steps=1000,
        neighbor=rl.TextNeighbor.NEUMANN)    
    agent = rl.SarsaAgent(
            states_size=env.states_size,
            actions_size=env.actions_size,
            alpha=0.1,
            epsilon=0.2,
            gamma=0.99
        )
    agent2 = rl.QLearningAgent(
            states_size=env.states_size,
            actions_size=env.actions_size,
            alpha=0.1,
            epsilon=0.03,
            gamma=0.99
        )
    agent3 = rl.ExpectedSarsaAgent(
            states_size=env.states_size,
            actions_size=env.actions_size,
            alpha=0.1,
            epsilon=0.03,
            gamma=0.99
        )
    test_session = rl.Session(env)    
    test_session.add_agent(agent)
    test_session.add_agent(agent2)
    test_session.add_agent(agent3)        
    test_session.reset_env()
    
    result = test_session.run(1000, 100, 50)
    rl.plot(result)
    
    out = []
    for episode in tqdm(range(100)):
        test_session.reset_env()
        out.append(
            test_session._run_test(episode, 50, False)
        )
    
    rl.plot(pd.concat(out))
    