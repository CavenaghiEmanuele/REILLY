#pragma once

#include "approximate_temporal_difference.ipp"

namespace reilly {

namespace agents {

class SemiGradientSarsa : public ApproximateTemporalDifference {
   protected:
    void update(Vector next_state, float reward, bool done, py::kwargs kwargs);

   public:
    SemiGradientSarsa(size_t actions, float alpha, float epsilon, float gamma, float epsilon_decay, py::kwargs kwargs);
    SemiGradientSarsa(const SemiGradientSarsa &other);
    SemiGradientSarsa &operator=(const SemiGradientSarsa &other);
    virtual ~SemiGradientSarsa();
};

}  // namespace agents

}  // namespace reilly