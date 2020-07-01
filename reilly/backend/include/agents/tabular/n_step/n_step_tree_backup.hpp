#pragma once

#include "n_step.ipp"

namespace reilly {

namespace agents {

class NStepTreeBackup : public NStep {
   public:
    NStepTreeBackup(size_t states, size_t actions, float alpha, float epsilon, float gamma, size_t n_step, float epsilon_decay = 1);
    NStepTreeBackup(const NStepTreeBackup &other);
    NStepTreeBackup &operator=(const NStepTreeBackup &other);
    virtual ~NStepTreeBackup();

    void update(size_t next_state, float reward, bool done, py::kwargs kwargs);
};

}  // namespace agents

}  // namespace reilly
