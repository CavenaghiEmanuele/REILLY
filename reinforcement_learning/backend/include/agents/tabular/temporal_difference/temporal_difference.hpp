#pragma once

#include "../../agent.ipp"

namespace rl {

namespace agents {

class TemporalDifference : public Agent {
   protected:
    float alpha;

   public:
    TemporalDifference(size_t states, size_t actions, float alpha, float epsilon, float gamma, float epsilon_decay);
    TemporalDifference(const TemporalDifference &other);
    virtual ~TemporalDifference();

    void reset(size_t init_state);

    std::string __repr__();
};

}  // namespace agents

}  // namespace rl
