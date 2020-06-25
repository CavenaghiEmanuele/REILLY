import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import reinforcement_learning as rl


def T():
    # 23 x 16
    env = rl.TextEnvironment(
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
        neighbor=rl.TextNeighbor.MOORE,
        raw_state=True
    )

    return env
