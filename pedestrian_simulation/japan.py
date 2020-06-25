import reinforcement_learning as rl


def japan():

    ENV = rl.TextEnvironment(
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

        max_steps=100,
        neighbor=rl.TextNeighbor.NEUMANN,
        raw_state=True
    )
