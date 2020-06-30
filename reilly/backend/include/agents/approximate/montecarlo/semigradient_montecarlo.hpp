#pragma once

#include "../approximate_agent.ipp"

namespace rl {

namespace agents {
class SemiGradientMonteCarlo : public ApproximateAgent {
   protected:
    Trajectory trajectory;

    void reset(Vector init_state);
    void update(Vector next_state, float reward, bool done, py::kwargs kwargs);

   public:
    SemiGradientMonteCarlo(size_t actions, float alpha, float epsilon, float gamma, float epsilon_decay,
                           size_t features, size_t tilings, std::list<float> tilings_offset,
                           std::list<float> tile_size);
    SemiGradientMonteCarlo(const SemiGradientMonteCarlo &other);
    SemiGradientMonteCarlo &operator=(const SemiGradientMonteCarlo &other);
    virtual ~SemiGradientMonteCarlo();
};
}  // namespace agents

}  // namespace rl