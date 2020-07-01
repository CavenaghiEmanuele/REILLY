#pragma once

#include "../approximate_agent.ipp"

namespace rl {

namespace agents {

class ApproximateTemporalDifference : public ApproximateAgent {
   protected:
    virtual void reset(Vector init_state);

   public:
    ApproximateTemporalDifference(size_t actions, float alpha, float epsilon, float gamma, float epsilon_decay,
                                  py::kwargs kwargs);
    ApproximateTemporalDifference(const ApproximateTemporalDifference &other);
    virtual ~ApproximateTemporalDifference();
};

}  // namespace agents

}  // namespace rl