#pragma once

#include "../approximate_agent.ipp"

namespace rl {

namespace agents {

class ApproximateNStep : public ApproximateAgent {
   protected:
    size_t n_step;
    size_t T;

    Trajectory trajectory;

    virtual void reset(Vector init_state);

   public:
    ApproximateNStep(size_t actions, float alpha, float epsilon, float gamma, size_t n_step, float epsilon_decay,
                     size_t features, size_t tilings, std::list<float> tilings_offset, std::list<float> tile_size);
    ApproximateNStep(const ApproximateNStep &other);
    virtual ~ApproximateNStep();

    std::string __repr__();
};

}  // namespace agents

}  // namespace rl