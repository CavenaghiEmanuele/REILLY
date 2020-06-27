#pragma once

#include "../agent.ipp"

namespace rl {

namespace agents {

using ActionValue = xt::xtensor<float, 2>;
using Policy = xt::xtensor<float, 2>;

class TabularAgent : public Agent {
   protected:
    size_t states;
    size_t actions;

    ActionValue Q;
    Policy pi;

    inline size_t argmaxQs(const ActionValue &Q, size_t state);
    inline virtual size_t select_action(const Policy &pi, size_t state);
    inline virtual void policy_update(const ActionValue &Q, Policy &pi, size_t state);

   public:
    TabularAgent(size_t states, size_t actions, float alpha, float epsilon, float gamma, float epsilon_decay = 1);
    TabularAgent(const TabularAgent &other);
    virtual ~TabularAgent();

    std::string __repr__();
};

}  // namespace agents

}  // namespace rl
