import reinforcement_learning as rl


def corridor():

    # 25 x 8
    ENV = rl.TextEnvironment(
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
        neighbor=rl.TextNeighbor.NEUMANN,
        raw_state=True
    )
    
