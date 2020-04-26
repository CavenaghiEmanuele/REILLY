import os
import matplotlib.pyplot as plt
from multiprocessing import Pool

import reinforcement_learning as rl


def run_agent(agent):
    return {agent: agent.run(n_episodes, n_tests, test_step)}

def build_agent(environment):

    print("*************************************************")
    print("*                    AGENT                      *")
    print("*************************************************")
    agent_type = input("Insert the agent type: ")

    if agent_type == "MonteCarlo" or agent_type == "MC":
        epsilon = float(input("Insert the parameter epsilon: "))
        gamma = float(input("Insert the parameter gamma: "))
        visit_update = input("Insert visit type (first, every): ")

        return rl.MonteCarloAgent(epsilon, gamma, environment, visit_update)

    if agent_type == "Sarsa" or agent_type == "S":
        alpha = float(input("Insert the parameter alpha (learning rate): "))
        epsilon = float(input("Insert the parameter epsilon: "))
        gamma = float(input("Insert the parameter gamma: "))

        return rl.SarsaAgent(alpha ,epsilon, gamma, environment)
    
    if agent_type == "ExpectedSarsa" or agent_type == "ES":
        alpha = float(input("Insert the parameter alpha (learning rate): "))
        epsilon = float(input("Insert the parameter epsilon: "))
        gamma = float(input("Insert the parameter gamma: ")) 

        return rl.ExpectedSarsaAgent(alpha, epsilon, gamma, environment)

    if agent_type == "Q learning" or agent_type == "QL":
        alpha = float(input("Insert the parameter alpha (learning rate): "))
        epsilon = float(input("Insert the parameter epsilon: "))
        gamma = float(input("Insert the parameter gamma: "))

        return rl.QLearningAgent(alpha, epsilon, gamma, environment)
    
    raise Exception("Agent doesn't exist")

def build_environment(env_name) -> rl.Environment:

    if env_name == "FrozenLake":
        return rl.Frozen_Lake4x4(slippery=True)
    if env_name == "FrozenLakeNotSlippery4x4":
        return rl.Frozen_Lake4x4(slippery=False)
    if env_name == "FrozenLake8x8":
        return rl.Frozen_Lake8x8(slippery=True)
    if env_name == "FrozenLakeNotSlippery8x8":
        return rl.Frozen_Lake8x8(slippery=False) 

    if env_name == "Taxi":
        return rl.Taxi()
    
    raise Exception("Environment doesn't exist")

def plot(env, results):
    tests_list = env.get_env_tests()
    legend = []
    for test in tests_list:
        plt.figure(test)
        
        for result in results:
            key = [key for key in result.keys()][0] #There is only one dict for every result
            legend.append(key)
            plt.plot(result[key][test])
        
        plt.legend(legend, loc='lower right')
        plt.ylabel(test)
        plt.xlabel("Number of tests")
        plt.grid(linestyle="--", linewidth=0.5, color='.25', zorder=-10)

    plt.show()


if __name__ == '__main__':

    global n_episodes
    global n_tests
    global test_step

    env_name = input("Insert the enviroment name: ")
    environment = build_environment(env_name) #Creazione ambiente

    n_episodes = int(input("Insert n° episodes of training: "))
    n_tests = int(input("Insert n° episodes for every test: "))
    test_step = int(input("Insert number of episodes between tests: "))

    agents_list = []
    n_agents = int(input("Insert the number of agents: "))
    for i in range(n_agents):
        agents_list.append(build_agent(environment))
           

    pool = Pool(len(os.sched_getaffinity(0))) #Creo un pool di processi
    results = pool.starmap(run_agent, zip(agents_list)) #Ogni agente viene affidato ad un processo
    pool.close()
    pool.join() #Attendo che tutti gli agenti abbiano terminato il training per poi proseguire

    plot(environment, results)
    