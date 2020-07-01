#pragma once

#include "temporal_difference.ipp"

namespace reilly {

namespace agents {

class ExpectedSarsa : public TemporalDifference {
   public:
    ExpectedSarsa(size_t states, size_t actions, float alpha, float epsilon, float gamma, float epsilon_decay = 1);
    ExpectedSarsa(const ExpectedSarsa &other);
    ExpectedSarsa &operator=(const ExpectedSarsa &other);
    virtual ~ExpectedSarsa();

    void update(size_t next_state, float reward, bool done, py::kwargs kwargs);
};

}  // namespace agents

}  // namespace reilly
