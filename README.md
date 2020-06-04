# Reinforcement Learning Library

## TABULAR AGENTS

### MonteCarlo

* [x] on-policy
    * [x] MonteCarlo prediction
        * [x] first visit implementation
        * [x] every visit implementation
    * [x] MonteCarlo Control
* [ ] off-policy

### Temporal Difference

* [x] **Sarsa**
* [x] **Q-learning**
* [ ] **Expected Sarsa**
    * [x] on-policy
    * [ ] off-policy

### Double Temporal Difference

* [x] **Double Sarsa**
* [x] **Double Q-Learning**
* [x] **Double Expected Sarsa**

### n-step Bootstrapping

* [ ] **n-step Sarsa**
    * [x] on-policy
    * [ ] off-policy
* [ ] **n-step Expected Sarsa**
    * [x] on-policy
    * [ ] off-policy
* [ ] **n-step Tree Backup Algorithm**
* [ ] **n-step Q(sigma)**

### Planning and learning with tabular

* [ ] **Random-sample one-step tabular Q-planning**
* [ ] **Tabular Dyna-Q**
* [ ] **Prioritized sweeping**



## APPROXIMATE AGENTS

### Tile coding
* [x] 1-D tilings
* [x] n-D tilings
* [x] tiling offset
* [x] different tiling dimensions
* [ ] test of tiling

### Q Estimator
* [x] Base implementation
* [ ] With trace

### Temporal difference
* [ ] **Semi-gradient Sarsa**
    * [ ] on-policy 
        * [ ] test
    * [ ] off-policy
        * [ ] test
    * [ ] differential
        * [ ] test    

* [ ] **Semi-gradient n-step Expected Sarsa**
    * [ ] on-policy
        * [ ] test
    * [ ] off-policy
        * [ ] test
    * [ ] differential
        * [ ] test

### n-step Bootstrapping
* [ ] **Semi-gradient n-step Sarsa**
    * [x] on-policy
        * [x] test
    * [ ] off-policy
        * [ ] test
    * [ ] differential
        * [ ] test

* [ ] **Semi-gradient n-step Expected Sarsa**
    * [ ] on-policy
        * [ ] test
    * [ ] off-policy
        * [ ] test
    * [ ] differential
        * [ ] test

    
## ENVIRONMENTS

* [ ] **GYM environments**
    * [x] FrozenLake4x4
    * [x] FrozenLake8x8
    * [x] Taxi
    * [x] MountainCar
    * [ ] ...

