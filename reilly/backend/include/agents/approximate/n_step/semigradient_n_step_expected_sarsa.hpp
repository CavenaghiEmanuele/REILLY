#pragma once

#include "approximate_n_step.hpp"

namespace reilly {

namespace agents {

class SemiGradientNStepExpectedSarsa : public ApproximateNStep {
   protected:
    void update(Vector next_state, float reward, bool done, py::kwargs kwargs);

   public:
    SemiGradientNStepExpectedSarsa(size_t actions, float alpha, float epsilon, float gamma, size_t n_step,
                                   float epsilon_decay, py::kwargs kwargs);
    SemiGradientNStepExpectedSarsa(const SemiGradientNStepExpectedSarsa &other);
    SemiGradientNStepExpectedSarsa &operator=(const SemiGradientNStepExpectedSarsa &other);
    virtual ~SemiGradientNStepExpectedSarsa();
};

}  // namespace agents

}  // namespace reilly