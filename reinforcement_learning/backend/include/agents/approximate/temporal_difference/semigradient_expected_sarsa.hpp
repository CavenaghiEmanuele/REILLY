#pragma once

#include "approximate_temporal_difference.ipp"

namespace rl {

namespace agents {

class SemiGradientExpectedSarsa : public ApproximateTemporalDifference {
   protected:
    void update(Vector next_state, float reward, bool done, py::kwargs kwargs);

   public:
    SemiGradientExpectedSarsa(size_t actions, float alpha, float epsilon, float gamma, float epsilon_decay,
                              size_t features, size_t tilings, std::list<float> tilings_offset,
                              std::list<float> tile_size);
    SemiGradientExpectedSarsa(const SemiGradientExpectedSarsa &other);
    SemiGradientExpectedSarsa &operator=(const SemiGradientExpectedSarsa &other);
    virtual ~SemiGradientExpectedSarsa();
};

}  // namespace agents

}  // namespace rl