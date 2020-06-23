import reinforcement_learning as rl


def rotated_corridor():

    # 25 x 8
    ENV = rl.TextEnvironment(
        text=   "                  #X######\n" +
                "                 #  X#####\n" +
                "                #    X####\n" +
                "               #      X###\n" +
                "              #        X##\n" +
                "             #          X#\n" +
                "            #            #\n" +
                "           #            # \n" +
                "          #            #  \n" +
                "         #            #   \n" +
                "        #            #    \n" +
                "       #            #     \n" +
                "      #            #      \n" +
                "     #            #       \n" +
                "    #            #        \n" +
                "   #            #         \n" +
                "  #            #          \n" +
                " #            #           \n" +
                "#S           #            \n" +
                "##S         #             \n" +
                "###S       #              \n" +
                "####S     #               \n" +
                "#####S   #                \n" +
                "######S #                 \n" +
                "########                  \n",
                
        max_steps=100,
        neighbor=rl.TextNeighbor.NEUMANN,
        raw_state=True
    )
    
