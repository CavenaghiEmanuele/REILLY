#pragma once

#include "approximate_n_step.hpp"

namespace rl {

namespace agents {

class SemiGradientNStepSarsa : public ApproximateNStep {
   protected:
    void update(Vector next_state, float reward, bool done, py::kwargs kwargs);

   public:
    SemiGradientNStepSarsa(size_t actions, float alpha, float epsilon, float gamma, size_t n_step, float epsilon_decay,
                           py::kwargs kwargs);
    SemiGradientNStepSarsa(const SemiGradientNStepSarsa &other);
    SemiGradientNStepSarsa &operator=(const SemiGradientNStepSarsa &other);
    virtual ~SemiGradientNStepSarsa();
};

}  // namespace agents

}  // namespace rl