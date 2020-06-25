import reinforcement_learning as rl


def three_room():

    ENV = rl.TextEnvironment(
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
                "##                                   ##\n" +
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
        neighbor=rl.TextNeighbor.NEUMANN,
        raw_state=True
    )
