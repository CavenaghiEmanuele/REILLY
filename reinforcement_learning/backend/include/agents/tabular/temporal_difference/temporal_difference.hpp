#pragma once

#include "../tabular_agent.ipp"

namespace rl {

namespace agents {

class TemporalDifference : public TabularAgent {
   public:
    TemporalDifference(size_t states, size_t actions, float alpha, float epsilon, float gamma, float epsilon_decay);
    TemporalDifference(const TemporalDifference &other);
    virtual ~TemporalDifference();

    virtual void reset(size_t init_state);
};

}  // namespace agents

}  // namespace rl
