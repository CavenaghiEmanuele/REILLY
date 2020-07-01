#pragma once

#include "approximate_temporal_difference.ipp"

namespace reilly {

namespace agents {

class SemiGradientExpectedSarsa : public ApproximateTemporalDifference {
   protected:
    void update(Vector next_state, float reward, bool done, py::kwargs kwargs);

   public:
    SemiGradientExpectedSarsa(size_t actions, float alpha, float epsilon, float gamma, float epsilon_decay,
                              py::kwargs kwargs);
    SemiGradientExpectedSarsa(const SemiGradientExpectedSarsa &other);
    SemiGradientExpectedSarsa &operator=(const SemiGradientExpectedSarsa &other);
    virtual ~SemiGradientExpectedSarsa();
};

}  // namespace agents

}  // namespace reilly