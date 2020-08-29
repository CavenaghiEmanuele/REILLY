<pre><b>
 ________  _______   ___  ___       ___           ___    ___ 
|\   __  \|\  ___ \ |\  \|\  \     |\  \         |\  \  /  /|
\ \  \|\  \ \   __/|\ \  \ \  \    \ \  \        \ \  \/  / /
 \ \   _  _\ \  \_|/_\ \  \ \  \    \ \  \        \ \    / / 
  \ \  \\  \\ \  \_|\ \ \  \ \  \____\ \  \____    \/  /  /  
   \ \__\\ _\\ \_______\ \__\ \_______\ \_______\__/  / /    
    \|__|\|__|\|_______|\|__|\|_______|\|_______|\___/ /     
                                                \|___|/      
                                                             
</b></pre>

[![Build Status](https://travis-ci.org/CavenaghiEmanuele/REILLY.svg?branch=master)](https://travis-ci.org/CavenaghiEmanuele/REILLY) [![CodeFactor](https://www.codefactor.io/repository/github/cavenaghiemanuele/reilly/badge)](https://www.codefactor.io/repository/github/cavenaghiemanuele/reilly) [![codecov](https://codecov.io/gh/CavenaghiEmanuele/REILLY/branch/master/graph/badge.svg)](https://codecov.io/gh/CavenaghiEmanuele/REILLY)

# Reinforcement Learning Library

## How to Install

Clone the repository including submodules:

    git clone --recurse-submodules -j8 https://github.com/CavenaghiEmanuele/REILLY.git

Build the package with C++ backend and install:

    cd REILLY && sudo python3 setup.py install

## Legends

* *empty* - Not implemented
* :heavy_check_mark: - Already implemented
* :x: - Non-existent
  
## Multi-Armed Bandits

### Bernoulli and Gaussian

| Name                                                                              |     Bernoulli      |      Gaussian      | Python |       C/C++        |
| --------------------------------------------------------------------------------- | :----------------: | :----------------: | :----: | :----------------: |
| Greedy                                                                            | :heavy_check_mark: | :heavy_check_mark: |        | :heavy_check_mark: |
| UCB                                                                               | :heavy_check_mark: | :heavy_check_mark: |        | :heavy_check_mark: |
| Thompson Samplig                                                                  | :heavy_check_mark: | :heavy_check_mark: |        | :heavy_check_mark: |
| [Dynamic Thompson Samplig](https://ieeexplore.ieee.org/abstract/document/6147024) | :heavy_check_mark: |                    |        | :heavy_check_mark: |
| [Discounted Thompson Samplig](https://arxiv.org/abs/1707.09727)                   | :heavy_check_mark: |                    |        | :heavy_check_mark: |

## Tabular Agents

### MonteCarlo

| Name                     |     On-Policy      | Off-Policy |       Python       |       C/C++        |
| ------------------------ | :----------------: | :--------: | :----------------: | :----------------: |
| MonteCarlo (First Visit) | :heavy_check_mark: |            | :heavy_check_mark: | :heavy_check_mark: |
| MonteCarlo (Every Visit) | :heavy_check_mark: |            | :heavy_check_mark: | :heavy_check_mark: |

### Temporal Difference

| Name           |     On-Policy      |     Off-Policy     |       Python       |       C/C++        |
| -------------- | :----------------: | :----------------: | :----------------: | :----------------: |
| Sarsa          | :heavy_check_mark: |                    | :heavy_check_mark: | :heavy_check_mark: |
| Q-learning     |        :x:         | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| Expected Sarsa | :heavy_check_mark: |                    | :heavy_check_mark: | :heavy_check_mark: |

### Double Temporal Difference

| Name                  |     On-Policy      |     Off-Policy     |       Python       |       C/C++        |
| --------------------- | :----------------: | :----------------: | :----------------: | :----------------: |
| Double Sarsa          | :heavy_check_mark: |                    | :heavy_check_mark: | :heavy_check_mark: |
| Double Q-learning     |        :x:         | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| Double Expected Sarsa | :heavy_check_mark: |                    | :heavy_check_mark: | :heavy_check_mark: |

### n-step Bootstrapping

| Name                  |     On-Policy      |     Off-Policy     |       Python       |       C/C++        |
| --------------------- | :----------------: | :----------------: | :----------------: | :----------------: |
| n-step Sarsa          | :heavy_check_mark: |                    | :heavy_check_mark: | :heavy_check_mark: |
| n-step Expected Sarsa | :heavy_check_mark: |                    | :heavy_check_mark: | :heavy_check_mark: |
| n-step Tree Backup    |        :x:         | :heavy_check_mark: |                    | :heavy_check_mark: |
| n-step Q(&sigma;)     |                    |                    |                    |                    |

### Planning and learning with tabular

| Name                                      | Python |       C/C++        |
| ----------------------------------------- | :----: | :----------------: |
| Random-sample one-step tabular Q-planning |        | :heavy_check_mark: |
| Tabular Dyna-Q                            |        | :heavy_check_mark: |
| Tabular Dyna-Q+                           |        | :heavy_check_mark: |
| Prioritized sweeping                      |        | :heavy_check_mark: |

## Approximate Agents

### Tile coding

| Name                        |       Python       |       C/C++        |
| --------------------------- | :----------------: | :----------------: |
| 1-D Tiling                  | :heavy_check_mark: | :heavy_check_mark: |
| n-D Tiling                  | :heavy_check_mark: | :heavy_check_mark: |
| Tiling offset               | :heavy_check_mark: | :heavy_check_mark: |
| Different tiling dimensions | :heavy_check_mark: | :heavy_check_mark: |

### Q Estimator

| Name                |       Python       |       C/C++        |
| ------------------- | :----------------: | :----------------: |
| Base implementation | :heavy_check_mark: | :heavy_check_mark: |
| With trace          | :heavy_check_mark: |                    |

### MonteCarlo

| Name                     |     On-Policy      | Off-Policy | Python |       C/C++        |
| ------------------------ | :----------------: | :--------: | :----: | :----------------: |
| Semi-gradient MonteCarlo | :heavy_check_mark: |            |        | :heavy_check_mark: |

### Temporal difference

| Name                         |     On-Policy      | Off-Policy | Differential |       Python       |       C/C++        |
| ---------------------------- | :----------------: | :--------: | :----------: | :----------------: | :----------------: |
| Semi-gradient Sarsa          | :heavy_check_mark: |            |              | :heavy_check_mark: | :heavy_check_mark: |
| Semi-gradient Expected Sarsa | :heavy_check_mark: |            |              | :heavy_check_mark: | :heavy_check_mark: |

### n-step Bootstrapping

| Name                                |     On-Policy      | Off-Policy | Differential |       Python       |       C/C++        |
| ----------------------------------- | :----------------: | :--------: | :----------: | :----------------: | :----------------: |
| Semi-gradient n-step Sarsa          | :heavy_check_mark: |            |              | :heavy_check_mark: | :heavy_check_mark: |
| Semi-gradient n-step Expected Sarsa | :heavy_check_mark: |            |              | :heavy_check_mark: | :heavy_check_mark: |

### Traces

| Name               |     On-Policy      | Off-Policy |       Python       | C/C++ |
| ------------------ | :----------------: | :--------: | :----------------: | :---: |
| Accumulating Trace | :heavy_check_mark: |            | :heavy_check_mark: |       |
| Replacing Trace    | :heavy_check_mark: |            | :heavy_check_mark: |       |
| Dutch Trace        |                    |            |                    |       |

### Eligibility Traces

| Name                           |     On-Policy      | Off-Policy |       Python       | C/C++ |
| :----------------------------- | :----------------: | :--------- | :----------------: | :---: |
| Temporal difference (&lambda;) |                    |            |                    |       |
| True Online TD(&lambda;)       |                    |            |                    |       |
| Sarsa(&lambda;)                | :heavy_check_mark: |            | :heavy_check_mark: |       |
| True Online Sarsa(&lambda;)    |                    |            |                    |       |
| Forward Sarsa(&lambda;)        |                    |            |                    |       |
| Watkins’s Q(&lambda;)          |                    |            |                    |       |
| Tree-Backup Q(&lambda;)        |                    |            |                    |       |

## Environments

### GYM Environments

| Name          | Discrete State? | Discrete Action? | Linear State? | Multi-Agent? |
| ------------- | :-------------: | :--------------: | :-----------: | :----------: |
| FrozenLake4x4 |       Yes       |       Yes        |      Yes      |      No      |
| FrozenLake8x8 |       Yes       |       Yes        |      Yes      |      No      |
| Taxi          |       Yes       |       Yes        |      Yes      |      No      |
| MountainCar   |       No        |       Yes        |      No       |      No      |

### Custom Environments


| Name | Discrete State? | Discrete Action? | Linear State? | Multi-Agent? |
| ---- | :-------------: | :--------------: | :-----------: | :----------: |
| Text |       Yes       |       Yes        |      No       |     Yes      |

## Sessions

| Name         | Multi-Agent? | Joint Train? | Joint Test? |
| ------------ | :----------: | :----------: | :---------: |
| Session      |      No      |      No      |     No      |
| JointSession |     Yes      |   Optional   |     Yes     |
