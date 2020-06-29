import reinforcement_learning as rl


def japan():

    env = rl.TextEnvironment(
        text="##########XXXXXX###\n" +
             "#                 #\n" +
             "#                 #\n" +
             "#                 #\n" +
             "#                 #\n" +
             "#                 #\n" +
             "#                 #\n" +
             "#                 #\n" +
             "#                 #\n" +
             "#                 #\n" +
             "#                 #\n" +
             "#                 #\n" +
             "#                 #\n" +
             "#                 #\n" +
             "#                 #\n" +
             "###  ##  #####  ###\n" +
             "#                 #\n" +
             "#                 #\n" +
             "#                 #\n" +
             "#                 #\n" +
             "#                 #\n" +
             "#                 #\n" +
             "#                 #\n" +
             "#                 #\n" +
             "#                 #\n" +
             "#                 #\n" +
             "#                 #\n" +
             "#                 #\n" +
             "#                 #\n" +
             "#                 #\n" +
             "##########SSSSSS###",

        max_steps=250,
        neighbor=rl.TextNeighbor.MOORE,
        raw_state=True
    )
    
    return env
