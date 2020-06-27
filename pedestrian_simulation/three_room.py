import reinforcement_learning as rl


def three_room():

    env = rl.TextEnvironment(
        text=   " #####XXX#########XXX#########XXX##### \n" +
                " #           #           #           # \n" +
                " #           #           #           # \n" +
                " #           #           #           # \n" +
                " #           #           #           # \n" +
                " #           #           #           # \n" +
                " #           #           #           # \n" +
                " #           #           #           # \n" +
                " #           #           #           # \n" +
                " #           #           #           # \n" +
                " #           #           #           # \n" +
                " #           #           #           # \n" +
                " #           #           #           # \n" +
                " #           #           #           # \n" +
                " #           #           #           # \n" +
                " #           #           #           # \n" +
                "##           #           #           ##\n" +
                "#S                                   S#\n" +
                "#S                                   S#\n" +
                "#S                                   S#\n" +
                "#S                                   S#\n" +
                "#S                                   S#\n" +
                "#S                                   S#\n" +
                "#S                                   S#\n" +
                "#S                                   S#\n" +
                "##           #           #           ##\n" +
                " #           #           #           # \n" +
                " #           #           #           # \n" +
                " #           #           #           # \n" +
                " #           #           #           # \n" +
                " #           #           #           # \n" +
                " #           #           #           # \n" +
                " #           #           #           # \n" +
                " #           #           #           # \n" +
                " #           #           #           # \n" +
                " #           #           #           # \n" +
                " #           #           #           # \n" +
                " #           #           #           # \n" +
                " #####XXX#########XXX#########XXX##### \n",

        max_steps=100,
        neighbor=rl.TextNeighbor.MOORE,
        raw_state=True
    )
    
    return env
