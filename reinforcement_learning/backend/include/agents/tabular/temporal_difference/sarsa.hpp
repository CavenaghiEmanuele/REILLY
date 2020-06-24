#pragma once

#include "temporal_difference.ipp"

namespace rl {

namespace agents {

class Sarsa : public TemporalDifference {
   public:
    Sarsa(size_t states, size_t actions, float alpha, float epsilon, float gamma, float epsilon_decay = 1);
    Sarsa(const Sarsa &other);
    Sarsa &operator=(const Sarsa &other);
    virtual ~Sarsa();

    void update(size_t next_state, float reward, bool done, py::kwargs kwargs);
};

}  // namespace agents

}  // namespace rl
