<pre><b>
 ___      _____
|    \   |       |  |      |     \   /
|___ /   |___    |  |      |      \ /
|    \   |       |  |      |       |
|     \  |_____  |  |_____ |_____  |  
</b></pre>

# Reinforcement Learning Library

Legends

* *empty* - Not implemented
* ✅ - Already implemented~~~~
* ❌ - Non existent

## Tabular Agents

### MonteCarlo


| Name | On-Policy | Off-Policy | Python | C/C++ |
| - | :-: | :-: | :-: | :-: |
| MonteCarlo (First Visit) | ✅ |   | ✅ |   |
| MonteCarlo (Every Visit) | ✅ |   | ✅ |   |

### Temporal Difference


| Name | On-Policy | Off-Policy | Python | C/C++ |
| - | :-: | :-: | :-: | :-: |
| Sarsa | ✅ |   | ✅ |   |
| Q-learning | ❌ | ✅ | ✅ |   |
| Expected Sarsa | ✅ |   | ✅ |   |

### Double Temporal Difference


| Name | On-Policy | Off-Policy | Python | C/C++ |
| - | :-: | :-: | :-: | :-: |
| Double Sarsa | ✅ |   | ✅ |   |
| Double Q-learning | ❌ | ✅ | ✅ |   |
| Double Expected Sarsa | ✅ |   | ✅ |   |

### n-step Bootstrapping


| Name | On-Policy | Off-Policy | Python | C/C++ |
| - | :-: | :-: | :-: | :-: |
| n-step Sarsa | ✅ |   | ✅ |   |
| n-step Expected Sarsa | ✅ |   | ✅ |   |
| n-step Tree Backup Algorithm |   |   |   |   |
| n-step Q$(\sigma)$ |   |   |   |   |

### Planning and learning with tabular


| Name | Python | C/C++ |
| - | :-: | :-: |
| Random-sample one-step tabular Q-planning |   |   |
| Tabular Dyna-Q |   |   |
| Prioritized sweeping |   |   |

## Approximate Agents

### Tile coding


| Name | Python | C/C++ |
| - | :-: | :-: |
| 1-D Tiling | ✅ |   |
| n-D Tiling | ✅ |   |
| Tiling offset | ✅ |   |
| Different tiling dimensions | ✅ |   |

### Q Estimator


| Name | Python | C/C++ |
| - | :-: | :-: |
| Base implementation | ✅ |   |
| With trace | ✅ |   |

### Temporal difference


| Name | On-Policy | Off-Policy | Differential | Python | C/C++ |
| - | :-: | :-: | :-: | :-: | :-: |
| Semi-gradient Sarsa | ✅ |   |   | ✅ |   |
| Semi-gradient Expected Sarsa | ✅ |   |   | ✅ |   |

### n-step Bootstrapping


| Name | On-Policy | Off-Policy | Differential | Python | C/C++ |
| - | :-: | :-: | :-: | :-: | :-: |
| Semi-gradient n-step Sarsa | ✅ |   |   | ✅ |   |
| Semi-gradient n-step Expected Sarsa | ✅ |   |   | ✅ |   |

### Traces


| Name | On-Policy | Off-Policy | Python | C/C++ |
| - | :-: | :-: | :-: | :-: |
| Accumulating Trace | ✅ |   | ✅ |   |
| Replacing Trace | ✅ |   | ✅ |   |
| Dutch Trace |   |   |   |   |

### Eligibility Traces


| Name | On-Policy | Off-Policy | Python | C/C++ |
| :- | :-: | :- | :-: | :-: |
| Temporal difference$(\lambda)$ |   |   |   |   |
| True Online TD$(\lambda)$ |   |   |   |   |
| Sarsa$(\lambda)$ | ✅ |   | ✅ |   |
| True Online Sarsa$(\lambda)$ |   |   |   |   |
| Forward Sarsa$(\lambda)$ |   |   |   |   |
| Watkins’s Q$(\lambda)$ |   |   |   |   |
| Tree-Backup Q$(\lambda)$ |   |   |   |   |

## Environments

### GYM Environments


| Name | Discrete State? | Discrete Action? | Linear State? | Multi-Agent? |
| - | :-: | :-: | :-: | :-: |
| FrozenLake4x4 | Yes | Yes | Yes | No |
| FrozenLake8x8 | Yes | Yes | Yes | No |
| Taxi | Yes | Yes | Yes | No |
| MountainCar | No | Yes | No | No |

### Custom Environments


| Name | Discrete State? | Discrete Action? | Linear State? | Multi-Agent? |
| - | :-: | :-: | :-: | :-: |
| Text | Yes | Yes | No | Yes |
