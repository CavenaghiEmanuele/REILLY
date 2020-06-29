#pragma once

#include "../approximate_agent.ipp"

namespace rl {

namespace agents {

class ApproximateTemporalDifference : public ApproximateAgent {
   protected:
    virtual void reset(State init_state);

   public:
    ApproximateTemporalDifference(size_t actions, float alpha, float epsilon, float gamma, float epsilon_decay,
                                  size_t tilings, std::list<float> tilings_offset, std::list<float> tile_size);
    ApproximateTemporalDifference(const ApproximateTemporalDifference &other);
    virtual ~ApproximateTemporalDifference();
};

}  // namespace agents

}  // namespace rl