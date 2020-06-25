#pragma once

#include "n_step.ipp"

namespace rl {

namespace agents {

class NStepSarsa : public NStep {
   public:
    NStepSarsa(size_t states, size_t actions, float alpha, float epsilon, float gamma, size_t n_step, float epsilon_decay = 1);
    NStepSarsa(const NStepSarsa &other);
    NStepSarsa &operator=(const NStepSarsa &other);
    virtual ~NStepSarsa();

    void update(size_t next_state, float reward, bool done, py::kwargs kwargs);
};

}  // namespace agents

}  // namespace rl
