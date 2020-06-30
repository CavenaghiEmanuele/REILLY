#pragma once

#include "approximate_temporal_difference.ipp"

namespace rl {

namespace agents {

class SemiGradientSarsa : public ApproximateTemporalDifference {
   protected:
    void update(Vector next_state, float reward, bool done, py::kwargs kwargs);

   public:
    SemiGradientSarsa(size_t actions, float alpha, float epsilon, float gamma, float epsilon_decay, size_t tilings,
                      size_t features, std::list<float> tilings_offset, std::list<float> tile_size);
    SemiGradientSarsa(const SemiGradientSarsa &other);
    SemiGradientSarsa &operator=(const SemiGradientSarsa &other);
    virtual ~SemiGradientSarsa();
};

}  // namespace agents

}  // namespace rl