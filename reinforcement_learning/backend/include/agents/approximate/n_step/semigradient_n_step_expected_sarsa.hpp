#pragma once

#include "approximate_n_step.hpp"

namespace rl {

namespace agents {

class SemiGradientNStepExpectedSarsa : public ApproximateNStep {
   protected:
    void update(Vector next_state, float reward, bool done, py::kwargs kwargs);

   public:
    SemiGradientNStepExpectedSarsa(size_t actions, float alpha, float epsilon, float gamma, size_t n_step, float epsilon_decay,
                           size_t tilings, size_t features, std::list<float> tilings_offset,
                           std::list<float> tile_size);
    SemiGradientNStepExpectedSarsa(const SemiGradientNStepExpectedSarsa &other);
    SemiGradientNStepExpectedSarsa &operator=(const SemiGradientNStepExpectedSarsa &other);
    virtual ~SemiGradientNStepExpectedSarsa();
};

}  // namespace agents

}  // namespace rl