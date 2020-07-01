#pragma once

#include "../approximate_agent.ipp"

namespace reilly {

namespace agents {

class ApproximateNStep : public ApproximateAgent {
   protected:
    size_t n_step;
    size_t T;

    Trajectory trajectory;

    virtual void reset(Vector init_state);

   public:
    ApproximateNStep(size_t actions, float alpha, float epsilon, float gamma, size_t n_step, float epsilon_decay,
                     py::kwargs kwargs);
    ApproximateNStep(const ApproximateNStep &other);
    virtual ~ApproximateNStep();

    std::string __repr__();
};

}  // namespace agents

}  // namespace reilly