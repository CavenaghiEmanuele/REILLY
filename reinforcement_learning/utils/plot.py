import matplotlib.pyplot as plt

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
