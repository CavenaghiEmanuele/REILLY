#pragma once

#include "double_temporal_difference.ipp"

namespace rl {

namespace agents {

class DoubleExpectedSarsa : public DoubleTemporalDifference {
   public:
    DoubleExpectedSarsa(size_t states, size_t actions, float alpha, float epsilon, float gamma, float epsilon_decay = 1);
    DoubleExpectedSarsa(const DoubleExpectedSarsa &other);
    DoubleExpectedSarsa &operator=(const DoubleExpectedSarsa &other);
    virtual ~DoubleExpectedSarsa();

    void update(size_t next_state, float reward, bool done, bool training);
};

}  // namespace agents

}  // namespace rl
