import reinforcement_learning as rl


def rotated_corridor():

    env = rl.TextEnvironment(
        text=   "                 ##X######\n" +
                "                ##  X#####\n" +
                "               ##    X####\n" +
                "              ##      X###\n" +
                "             ##        X##\n" +
                "            ##          X#\n" +
                "           ##            #\n" +
                "          ##            ##\n" +
                "         ##            ## \n" +
                "        ##            ##  \n" +
                "       ##            ##   \n" +
                "      ##            ##    \n" +
                "     ##            ##     \n" +
                "    ##            ##      \n" +
                "   ##            ##       \n" +
                "  ##            ##        \n" +
                " ##            ##         \n" +
                "##            ##          \n" +
                "#S           ##           \n" +
                "##S         ##            \n" +
                "###S       ##             \n" +
                "####S     ##              \n" +
                "#####S   ##               \n" +
                "######S ##                \n" +
                "#########                 \n",
                
        max_steps=250,
        neighbor=rl.TextNeighbor.MOORE,
        raw_state=True
    )

    return env
