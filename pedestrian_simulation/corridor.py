import reinforcement_learning as rl


def corridor():

    # 25 x 8
    env = rl.TextEnvironment(
        text=   "##########################\n" +
                "#S                       X\n" +
                "#S                       X\n" +
                "#S                       X\n" +
                "#S                       X\n" +
                "#S                       X\n" +
                "#S                       X\n" +
                "#S                       X\n" +
                "#S                       X\n" +
                "##########################\n",

        max_steps=100,
        neighbor=rl.TextNeighbor.MOORE,
        raw_state=True
    )
    
    return env
