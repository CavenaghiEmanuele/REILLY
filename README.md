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

# Reinforcement Learning Library

## How to Install

Clone the repository including submodules:

    git clone --recurse-submodules -j8 git@github.com:CavenaghiEmanuele/REILLY.git

Install requirements:

    pip3 install -r requirements.txt

Build the package and C++ backend:

    (sudo) python3 setup.py install

## Legends

* *empty* - Not implemented
* &#x2705; - Already implemented
* :x: - Non existent

## Tabular Agents

### MonteCarlo

| Name                     | On-Policy | Off-Policy |  Python  |  C/C++   |
| ------------------------ | :-------: | :--------: | :------: | :------: |
| MonteCarlo (First Visit) | &#x2705;  |            | &#x2705; | &#x2705; |
| MonteCarlo (Every Visit) | &#x2705;  |            | &#x2705; | &#x2705; |

### Temporal Difference

| Name           | On-Policy | Off-Policy |  Python  |  C/C++   |
| -------------- | :-------: | :--------: | :------: | :------: |
| Sarsa          | &#x2705;  |            | &#x2705; | &#x2705; |
| Q-learning     |    :x:    |  &#x2705;  | &#x2705; | &#x2705; |
| Expected Sarsa | &#x2705;  |            | &#x2705; | &#x2705; |

### Double Temporal Difference

| Name                  | On-Policy | Off-Policy |  Python  |  C/C++   |
| --------------------- | :-------: | :--------: | :------: | :------: |
| Double Sarsa          | &#x2705;  |            | &#x2705; | &#x2705; |
| Double Q-learning     |    :x:    |  &#x2705;  | &#x2705; | &#x2705; |
| Double Expected Sarsa | &#x2705;  |            | &#x2705; | &#x2705; |

### n-step Bootstrapping

| Name                  | On-Policy | Off-Policy |  Python  |  C/C++   |
| --------------------- | :-------: | :--------: | :------: | :------: |
| n-step Sarsa          | &#x2705;  |            | &#x2705; | &#x2705; |
| n-step Expected Sarsa | &#x2705;  |            | &#x2705; | &#x2705; |
| n-step Tree Backup    |    :x:    |  &#x2705;  |          | &#x2705; |
| n-step Q(&sigma;)     |           |            |          |          |

### Planning and learning with tabular

| Name                                      | Python |  C/C++   |
| ----------------------------------------- | :----: | :------: |
| Random-sample one-step tabular Q-planning |        | &#x2705; |
| Tabular Dyna-Q                            |        | &#x2705; |
| Tabular Dyna-Q+                           |        | &#x2705; |
| Prioritized sweeping                      |        | &#x2705; |

## Approximate Agents

### Tile coding

| Name                        |  Python  |  C/C++   |
| --------------------------- | :------: | :------: |
| 1-D Tiling                  | &#x2705; | &#x2705; |
| n-D Tiling                  | &#x2705; | &#x2705; |
| Tiling offset               | &#x2705; | &#x2705; |
| Different tiling dimensions | &#x2705; | &#x2705; |

### Q Estimator

| Name                |  Python  |  C/C++   |
| ------------------- | :------: | :------: |
| Base implementation | &#x2705; | &#x2705; |
| With trace          |          |          |

### MonteCarlo

| Name                     | On-Policy | Off-Policy | Python |  C/C++   |
| ------------------------ | :-------: | :--------: | :----: | :------: |
| Semi-gradient MonteCarlo | &#x2705;  |            |        | &#x2705; |

### Temporal difference

| Name                         | On-Policy | Off-Policy | Differential |  Python  |  C/C++   |
| ---------------------------- | :-------: | :--------: | :----------: | :------: | :------: |
| Semi-gradient Sarsa          | &#x2705;  |            |              | &#x2705; | &#x2705; |
| Semi-gradient Expected Sarsa | &#x2705;  |            |              | &#x2705; | &#x2705; |

### n-step Bootstrapping

| Name                                | On-Policy | Off-Policy | Differential |  Python  |  C/C++   |
| ----------------------------------- | :-------: | :--------: | :----------: | :------: | :------: |
| Semi-gradient n-step Sarsa          | &#x2705;  |            |              | &#x2705; | &#x2705; |
| Semi-gradient n-step Expected Sarsa | &#x2705;  |            |              | &#x2705; | &#x2705; |

### Traces

| Name               | On-Policy | Off-Policy |  Python  | C/C++ |
| ------------------ | :-------: | :--------: | :------: | :---: |
| Accumulating Trace | &#x2705;  |            | &#x2705; |       |
| Replacing Trace    | &#x2705;  |            | &#x2705; |       |
| Dutch Trace        |           |            |          |       |

### Eligibility Traces

| Name                           | On-Policy | Off-Policy |  Python  | C/C++ |
| :----------------------------- | :-------: | :--------- | :------: | :---: |
| Temporal difference (&lambda;) |           |            |          |       |
| True Online TD(&lambda;)       |           |            |          |       |
| Sarsa(&lambda;)                | &#x2705;  |            | &#x2705; |       |
| True Online Sarsa(&lambda;)    |           |            |          |       |
| Forward Sarsa(&lambda;)        |           |            |          |       |
| Watkins’s Q(&lambda;)          |           |            |          |       |
| Tree-Backup Q(&lambda;)        |           |            |          |       |

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
