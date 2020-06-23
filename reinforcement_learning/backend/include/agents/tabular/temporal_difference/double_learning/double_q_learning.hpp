#pragma once

#include "double_temporal_difference.ipp"

namespace rl {

namespace agents {

class DoubleQLearning : public DoubleTemporalDifference {
   public:
    DoubleQLearning(size_t states, size_t actions, float alpha, float epsilon, float gamma, float epsilon_decay = 1);
    DoubleQLearning(const DoubleQLearning &other);
    DoubleQLearning &operator=(const DoubleQLearning &other);
    virtual ~DoubleQLearning();

    void update(size_t next_state, float reward, bool done, bool training);
};

}  // namespace agents

}  // namespace rl
