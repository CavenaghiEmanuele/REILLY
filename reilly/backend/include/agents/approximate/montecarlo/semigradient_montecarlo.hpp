#pragma once

#include "../approximate_agent.ipp"

namespace reilly {

namespace agents {
class SemiGradientMonteCarlo : public ApproximateAgent {
   protected:
    Trajectory trajectory;

    void reset(Vector init_state);
    void update(Vector next_state, float reward, bool done, py::kwargs kwargs);

   public:
    SemiGradientMonteCarlo(size_t actions, float alpha, float epsilon, float gamma, float epsilon_decay,
                           py::kwargs kwargs);
    SemiGradientMonteCarlo(const SemiGradientMonteCarlo &other);
    SemiGradientMonteCarlo &operator=(const SemiGradientMonteCarlo &other);
    virtual ~SemiGradientMonteCarlo();
};
}  // namespace agents

}  // namespace reilly