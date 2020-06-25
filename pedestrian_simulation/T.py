import reinforcement_learning as rl


def T():

    # 23 x 16
    ENV = rl.TextEnvironment(
        text=   "       #XXXXXX#          \n" +
                "       #      #          \n" +
                "       #      #          \n" +
                "       #      #          \n" +
                "       #      #          \n" +
                "       #      #          \n" +
                "       #      #          \n" +
                "       #      #          \n" +
                "       #      #          \n" +
                "       #      #          \n" +
                "########      ###########\n" +
                "#S                     S#\n" +
                "#S                     S#\n" +
                "#S                     S#\n" +
                "#S                     S#\n" +
                "#S                     S#\n" +
                "#S                     S#\n" +
                "#########################\n",

        max_steps=100,
        neighbor=rl.TextNeighbor.NEUMANN,
        raw_state=True
    )
