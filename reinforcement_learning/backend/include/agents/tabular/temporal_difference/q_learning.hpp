#pragma once

#include "temporal_difference.ipp"

namespace rl {

namespace agents {

class QLearning : public TemporalDifference {
   public:
    QLearning(size_t states, size_t actions, float alpha, float epsilon, float gamma, float epsilon_decay = 1);
    QLearning(const QLearning &other);
    QLearning &operator=(const QLearning &other);
    virtual ~QLearning();

    void update(size_t next_state, float reward, bool done, py::kwargs kwargs);
};

}  // namespace agents

}  // namespace rl
